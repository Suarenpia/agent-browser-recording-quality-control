use serde_json::{json, Value};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::sync::oneshot;

use super::cdp::client::CdpClient;
use super::cdp::types::{CaptureScreenshotParams, CaptureScreenshotResult};

const DEFAULT_CAPTURE_FPS: u32 = 10;
const DEFAULT_SCREENSHOT_QUALITY: u8 = 80;
const DEFAULT_WEBM_CRF: u8 = 30;
const DEFAULT_WEBM_BITRATE: &str = "1M";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordingCodec {
    H264,
    Vp8,
    Vp9,
}

impl RecordingCodec {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "h264" => Ok(Self::H264),
            "vp8" => Ok(Self::Vp8),
            "vp9" => Ok(Self::Vp9),
            _ => Err(format!("Invalid recording codec: {}", value)),
        }
    }

    fn for_output_path(output_path: &str, requested: Option<Self>) -> Self {
        if let Some(codec) = requested {
            return codec;
        }
        if output_path.ends_with(".webm") {
            return Self::Vp8;
        }
        Self::H264
    }
}

/// Options from `record start` and `record restart` that affect capture and encoding quality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordingConfig {
    pub fps: u32,
    pub quality: u8,
    pub bitrate: Option<String>,
    pub crf: Option<u8>,
    pub codec: Option<RecordingCodec>,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            fps: DEFAULT_CAPTURE_FPS,
            quality: DEFAULT_SCREENSHOT_QUALITY,
            bitrate: None,
            crf: None,
            codec: None,
        }
    }
}

impl RecordingConfig {
    pub fn from_value(value: Option<&Value>) -> Result<Self, String> {
        let mut config = Self::default();
        let Some(options) = value else {
            return Ok(config);
        };

        if let Some(fps) = options.get("fps").and_then(|v| v.as_u64()) {
            if fps == 0 || fps > 60 {
                return Err("Recording fps must be between 1 and 60".to_string());
            }
            config.fps = fps as u32;
        }
        if let Some(quality) = options.get("quality").and_then(|v| v.as_u64()) {
            if quality > 100 {
                return Err("Recording quality must be between 0 and 100".to_string());
            }
            config.quality = quality as u8;
        }
        if let Some(bitrate) = options.get("bitrate").and_then(|v| v.as_str()) {
            config.bitrate = Some(bitrate.to_string());
        }
        if let Some(crf) = options.get("crf").and_then(|v| v.as_u64()) {
            if crf > 63 {
                return Err("Recording CRF must be between 0 and 63".to_string());
            }
            config.crf = Some(crf as u8);
        }
        if let Some(codec) = options.get("codec").and_then(|v| v.as_str()) {
            config.codec = Some(RecordingCodec::parse(codec)?);
        }

        Ok(config)
    }

    fn frame_interval(&self) -> Duration {
        Duration::from_micros((1_000_000 / self.fps as u64).max(1))
    }
}

pub struct RecordingState {
    pub active: bool,
    pub output_path: String,
    pub config: RecordingConfig,
    pub frame_count: u64,
    pub capture_task: Option<tokio::task::JoinHandle<Result<(), String>>>,
    pub shared_frame_count: Option<Arc<AtomicU64>>,
    pub cancel_tx: Option<oneshot::Sender<()>>,
}

impl RecordingState {
    pub fn new() -> Self {
        Self {
            active: false,
            output_path: String::new(),
            config: RecordingConfig::default(),
            frame_count: 0,
            capture_task: None,
            shared_frame_count: None,
            cancel_tx: None,
        }
    }
}

pub fn recording_start(
    state: &mut RecordingState,
    path: &str,
    config: RecordingConfig,
) -> Result<Value, String> {
    if state.active {
        return Err("Recording already active".to_string());
    }

    state.active = true;
    state.output_path = path.to_string();
    state.config = config;
    state.frame_count = 0;

    Ok(json!({ "started": true, "path": path }))
}

pub fn recording_stop(state: &mut RecordingState) -> Result<Value, String> {
    if !state.active {
        return Err("No recording in progress".to_string());
    }

    state.active = false;

    if state.frame_count == 0 {
        return Err("No frames captured".to_string());
    }

    Ok(json!({ "path": &state.output_path, "frames": state.frame_count }))
}

pub fn recording_restart(
    state: &mut RecordingState,
    path: &str,
    config: RecordingConfig,
) -> Result<Value, String> {
    let previous = if state.active {
        let stop_result = recording_stop(state);
        stop_result
            .ok()
            .and_then(|v| v.get("path").and_then(|p| p.as_str()).map(String::from))
    } else {
        None
    };

    recording_start(state, path, config)?;

    Ok(json!({
        "restarted": true,
        "previousPath": previous,
        "path": path,
    }))
}

fn build_ffmpeg_command(output_path: &str, config: &RecordingConfig) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new("ffmpeg");

    cmd.args(["-y"])
        .args(["-avioflags", "direct"])
        .args([
            "-fpsprobesize",
            "0",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
        ])
        .args([
            "-f",
            "image2pipe",
            "-c:v",
            "mjpeg",
            "-framerate",
            &config.fps.to_string(),
            "-i",
            "pipe:0",
        ])
        .args(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]);

    match RecordingCodec::for_output_path(output_path, config.codec) {
        RecordingCodec::H264 => {
            cmd.args(["-c:v", "libx264", "-preset", "ultrafast"]);
            if let Some(crf) = config.crf {
                cmd.args(["-crf", &crf.to_string()]);
            }
            if let Some(bitrate) = &config.bitrate {
                cmd.args(["-b:v", bitrate]);
            }
        }
        RecordingCodec::Vp8 => {
            let crf = config.crf.unwrap_or(DEFAULT_WEBM_CRF).to_string();
            let bitrate = config.bitrate.as_deref().unwrap_or(DEFAULT_WEBM_BITRATE);
            cmd.args(["-c:v", "libvpx", "-crf", &crf, "-b:v", bitrate]);
        }
        RecordingCodec::Vp9 => {
            let crf = config.crf.unwrap_or(DEFAULT_WEBM_CRF).to_string();
            let bitrate = config.bitrate.as_deref().unwrap_or(DEFAULT_WEBM_BITRATE);
            cmd.args(["-c:v", "libvpx-vp9", "-crf", &crf, "-b:v", bitrate]);
        }
    }

    cmd.args(["-pix_fmt", "yuv420p", "-threads", "1"])
        .arg(output_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    cmd
}

/// Spawn a background task that captures screenshots at a fixed interval
/// and pipes them to ffmpeg in real-time.
pub fn spawn_recording_task(
    client: Arc<CdpClient>,
    session_id: String,
    output_path: String,
    config: RecordingConfig,
    shared_count: Arc<AtomicU64>,
    cancel_rx: oneshot::Receiver<()>,
) -> tokio::task::JoinHandle<Result<(), String>> {
    tokio::spawn(async move {
        let mut cancel_rx = std::pin::pin!(cancel_rx);

        let mut ffmpeg = build_ffmpeg_command(&output_path, &config)
            .spawn()
            .map_err(|e| {
                format!(
                "ffmpeg not found or failed to execute: {}. Install ffmpeg to enable recording.",
                e
            )
            })?;

        let mut stdin = ffmpeg
            .stdin
            .take()
            .ok_or_else(|| "Failed to open ffmpeg stdin".to_string())?;

        let mut interval = tokio::time::interval(config.frame_interval());
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let params = CaptureScreenshotParams {
            format: Some("jpeg".to_string()),
            quality: Some(config.quality.into()),
            clip: None,
            from_surface: Some(true),
            capture_beyond_viewport: None,
        };

        loop {
            tokio::select! {
                _ = &mut cancel_rx => break,
                _ = interval.tick() => {}
            }

            let result: Result<CaptureScreenshotResult, _> = client
                .send_command_typed("Page.captureScreenshot", &params, Some(&session_id))
                .await;

            let screenshot = match result {
                Ok(s) => s,
                Err(e) => {
                    if e.contains("Target closed") || e.contains("not found") {
                        break;
                    }
                    continue;
                }
            };

            let bytes = match base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                &screenshot.data,
            ) {
                Ok(b) => b,
                Err(_) => continue,
            };

            if stdin.write_all(&bytes).await.is_err() {
                break;
            }
            shared_count.fetch_add(1, Ordering::Relaxed);
        }

        drop(stdin);

        let output = ffmpeg
            .wait_with_output()
            .await
            .map_err(|e| format!("ffmpeg wait failed: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "ffmpeg failed: {}",
                stderr.chars().take(300).collect::<String>()
            ));
        }

        Ok(())
    })
}

pub async fn stop_recording_task(state: &mut RecordingState) -> Result<(), String> {
    if let Some(tx) = state.cancel_tx.take() {
        let _ = tx.send(());
    }

    let counter = state.shared_frame_count.take();
    let handle = state.capture_task.take();

    let result = if let Some(h) = handle {
        match h.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(e) => Err(format!("Recording task panicked: {}", e)),
        }
    } else {
        Ok(())
    };

    if let Some(c) = counter {
        state.frame_count = c.load(Ordering::Relaxed);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recording_state_new() {
        let state = RecordingState::new();
        assert!(!state.active);
        assert!(state.output_path.is_empty());
        assert_eq!(state.frame_count, 0);
    }

    #[test]
    fn test_recording_start_sets_active() {
        let mut state = RecordingState::new();
        let result = recording_start(&mut state, "/tmp/test.mp4", RecordingConfig::default());
        assert!(result.is_ok());
        assert!(state.active);
        assert_eq!(state.output_path, "/tmp/test.mp4");
        assert_eq!(state.frame_count, 0);
    }

    #[test]
    fn test_recording_start_while_active() {
        let mut state = RecordingState::new();
        recording_start(&mut state, "/tmp/test1.mp4", RecordingConfig::default()).unwrap();
        let result = recording_start(&mut state, "/tmp/test2.mp4", RecordingConfig::default());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already active"));
    }

    #[test]
    fn test_recording_stop_not_active() {
        let mut state = RecordingState::new();
        let result = recording_stop(&mut state);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No recording"));
    }

    #[test]
    fn test_recording_stop_no_frames() {
        let mut state = RecordingState::new();
        recording_start(&mut state, "/tmp/test.mp4", RecordingConfig::default()).unwrap();
        let result = recording_stop(&mut state);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No frames"));
        assert!(!state.active);
    }

    #[test]
    fn test_recording_restart_while_inactive() {
        let mut state = RecordingState::new();
        let result = recording_restart(&mut state, "/tmp/new.webm", RecordingConfig::default());
        assert!(result.is_ok());
        assert!(state.active);
        assert_eq!(state.output_path, "/tmp/new.webm");
    }

    #[test]
    fn test_recording_restart_while_active() {
        let mut state = RecordingState::new();
        recording_start(&mut state, "/tmp/old.webm", RecordingConfig::default()).unwrap();
        state.frame_count = 10;
        let result =
            recording_restart(&mut state, "/tmp/new.webm", RecordingConfig::default()).unwrap();
        assert!(state.active);
        assert_eq!(state.output_path, "/tmp/new.webm");
        assert_eq!(state.frame_count, 0);
        assert_eq!(result["previousPath"], "/tmp/old.webm");
    }

    #[test]
    fn test_build_ffmpeg_command_webm() {
        let cmd = build_ffmpeg_command("/tmp/out.webm", &RecordingConfig::default());
        let args: Vec<&std::ffi::OsStr> = cmd.as_std().get_args().collect();
        let args_str: Vec<&str> = args.iter().filter_map(|a| a.to_str()).collect();
        assert!(args_str.contains(&"libvpx"));
        assert!(args_str.contains(&"/tmp/out.webm"));
    }

    #[test]
    fn test_build_ffmpeg_command_mp4() {
        let config = RecordingConfig {
            fps: 30,
            quality: 95,
            bitrate: Some("6M".to_string()),
            crf: Some(16),
            codec: Some(RecordingCodec::H264),
        };
        let cmd = build_ffmpeg_command("/tmp/out.mp4", &config);
        let args: Vec<&std::ffi::OsStr> = cmd.as_std().get_args().collect();
        let args_str: Vec<&str> = args.iter().filter_map(|a| a.to_str()).collect();
        assert!(args_str.contains(&"libx264"));
        assert!(args_str.contains(&"30"));
        assert!(args_str.contains(&"16"));
        assert!(args_str.contains(&"6M"));
        assert!(args_str.contains(&"/tmp/out.mp4"));
    }
}
