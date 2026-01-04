//! Debug module for Palace
//!
//! Provides a Unix socket server for debugging commands like screenshots.
//! Connect with: `nc -U /tmp/palace-debug.sock` or `socat - UNIX-CONNECT:/tmp/palace-debug.sock`
//!
//! Commands:
//! - `screenshot` - Capture current frame to /tmp/palace-screenshot-<timestamp>.png
//! - `screenshot <path>` - Capture current frame to specified path
//! - `help` - Show available commands
//! - `quit` - Close connection

use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;

const SOCKET_PATH: &str = "/tmp/palace-debug.sock";

/// Commands that can be sent to the renderer
#[derive(Debug)]
pub enum DebugCommand {
    Screenshot { path: Option<PathBuf> },
}

/// Response from the renderer
#[derive(Debug)]
pub enum DebugResponse {
    ScreenshotSaved { path: PathBuf },
    ScreenshotPending { path: PathBuf },
    Error { message: String },
}

/// Debug server handle
pub struct DebugServer {
    command_tx: mpsc::Sender<(DebugCommand, mpsc::Sender<DebugResponse>)>,
}

impl DebugServer {
    /// Start the debug server with an event loop proxy (event-driven, no polling)
    pub fn start_with_proxy(
        proxy: winit::event_loop::EventLoopProxy<crate::app::AppEvent>,
    ) -> Self {
        let (command_tx, _) = mpsc::channel(16);

        let server = Self {
            command_tx: command_tx.clone(),
        };

        // Spawn the socket listener with proxy
        tokio::spawn(async move {
            if let Err(e) = run_socket_server_with_proxy(proxy).await {
                tracing::error!("Debug server error: {}", e);
            }
        });

        server
    }

    /// Start the debug server in a background task (legacy, uses channel)
    #[allow(dead_code)]
    pub fn start() -> (Self, mpsc::Receiver<(DebugCommand, mpsc::Sender<DebugResponse>)>) {
        let (command_tx, command_rx) = mpsc::channel(16);

        let server = Self {
            command_tx: command_tx.clone(),
        };

        // Spawn the socket listener
        tokio::spawn(async move {
            if let Err(e) = run_socket_server(command_tx).await {
                tracing::error!("Debug server error: {}", e);
            }
        });

        (server, command_rx)
    }

    /// Get a sender for commands (used internally)
    #[allow(dead_code)]
    pub fn command_sender(&self) -> mpsc::Sender<(DebugCommand, mpsc::Sender<DebugResponse>)> {
        self.command_tx.clone()
    }
}

/// Event-driven socket server using EventLoopProxy
async fn run_socket_server_with_proxy(
    proxy: winit::event_loop::EventLoopProxy<crate::app::AppEvent>,
) -> anyhow::Result<()> {
    // Remove existing socket file
    let _ = std::fs::remove_file(SOCKET_PATH);

    let listener = UnixListener::bind(SOCKET_PATH)?;
    tracing::info!("Debug server listening on {}", SOCKET_PATH);

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let proxy = proxy.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_client_with_proxy(stream, proxy).await {
                        tracing::debug!("Client disconnected: {}", e);
                    }
                });
            }
            Err(e) => {
                tracing::error!("Failed to accept connection: {}", e);
            }
        }
    }
}

/// Handle client using EventLoopProxy
async fn handle_client_with_proxy(
    stream: UnixStream,
    proxy: winit::event_loop::EventLoopProxy<crate::app::AppEvent>,
) -> anyhow::Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    // Send welcome message
    writer
        .write_all(b"Palace Debug Server\nType 'help' for commands\n> ")
        .await?;

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break; // Client disconnected
        }

        let input = line.trim();
        if input.is_empty() {
            writer.write_all(b"> ").await?;
            continue;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        let response = match parts.as_slice() {
            ["help"] => {
                "Commands:\n  screenshot [path] - Capture screenshot (non-blocking)\n  getstate - Get current app state as JSON\n  kill - Terminate Palace\n  help - Show this help\n  quit - Close connection\n".to_string()
            }
            ["quit"] | ["exit"] => {
                writer.write_all(b"Goodbye!\n").await?;
                break;
            }
            ["kill"] | ["shutdown"] => {
                let pid = std::process::id();
                writer.write_all(format!("PID:{}\n", pid).as_bytes()).await?;
                let _ = proxy.send_event(crate::app::AppEvent::Shutdown);
                break;
            }
            ["screenshot"] => {
                execute_command_with_proxy(&proxy, DebugCommand::Screenshot { path: None }).await
            }
            ["screenshot", path] => {
                execute_command_with_proxy(
                    &proxy,
                    DebugCommand::Screenshot {
                        path: Some(PathBuf::from(path)),
                    },
                )
                .await
            }
            ["getstate"] => {
                get_state_from_app(&proxy).await
            }
            _ => format!("Unknown command: {}\nType 'help' for available commands\n", input),
        };

        writer.write_all(response.as_bytes()).await?;
        writer.write_all(b"> ").await?;
    }

    Ok(())
}

/// Get current app state via EventLoopProxy
async fn get_state_from_app(
    proxy: &winit::event_loop::EventLoopProxy<crate::app::AppEvent>,
) -> String {
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();

    if proxy
        .send_event(crate::app::AppEvent::GetState(response_tx))
        .is_err()
    {
        return "Error: Event loop not responding\n".to_string();
    }

    match tokio::time::timeout(std::time::Duration::from_secs(2), response_rx).await {
        Ok(Ok(state_json)) => format!("{}\n", state_json),
        Ok(Err(_)) => "Error: Channel closed\n".to_string(),
        Err(_) => "Error: Timeout waiting for state\n".to_string(),
    }
}

/// Execute command by sending through EventLoopProxy
async fn execute_command_with_proxy(
    proxy: &winit::event_loop::EventLoopProxy<crate::app::AppEvent>,
    command: DebugCommand,
) -> String {
    let (response_tx, mut response_rx) = mpsc::channel(1);

    if proxy
        .send_event(crate::app::AppEvent::DebugCommand(command, response_tx))
        .is_err()
    {
        return "Error: Event loop not responding\n".to_string();
    }

    match tokio::time::timeout(std::time::Duration::from_secs(5), response_rx.recv()).await {
        Ok(Some(DebugResponse::ScreenshotSaved { path })) => {
            format!("Screenshot saved: {}\n", path.display())
        }
        Ok(Some(DebugResponse::ScreenshotPending { path })) => {
            format!("Screenshot capture started: {}\n", path.display())
        }
        Ok(Some(DebugResponse::Error { message })) => {
            format!("Error: {}\n", message)
        }
        Ok(None) => "Error: No response from renderer\n".to_string(),
        Err(_) => "Error: Timeout waiting for response\n".to_string(),
    }
}

#[allow(dead_code)]
async fn run_socket_server(
    command_tx: mpsc::Sender<(DebugCommand, mpsc::Sender<DebugResponse>)>,
) -> anyhow::Result<()> {
    // Remove existing socket file
    let _ = std::fs::remove_file(SOCKET_PATH);

    let listener = UnixListener::bind(SOCKET_PATH)?;
    tracing::info!("Debug server listening on {}", SOCKET_PATH);

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let tx = command_tx.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_client(stream, tx).await {
                        tracing::debug!("Client disconnected: {}", e);
                    }
                });
            }
            Err(e) => {
                tracing::error!("Failed to accept connection: {}", e);
            }
        }
    }
}

#[allow(dead_code)]
async fn handle_client(
    stream: UnixStream,
    command_tx: mpsc::Sender<(DebugCommand, mpsc::Sender<DebugResponse>)>,
) -> anyhow::Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    // Send welcome message
    writer
        .write_all(b"Palace Debug Server\nType 'help' for commands\n> ")
        .await?;

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break; // Client disconnected
        }

        let input = line.trim();
        if input.is_empty() {
            writer.write_all(b"> ").await?;
            continue;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        let response = match parts.as_slice() {
            ["help"] => {
                "Commands:\n  screenshot [path] - Capture screenshot (non-blocking)\n  help - Show this help\n  quit - Close connection\n".to_string()
            }
            ["quit"] | ["exit"] => {
                writer.write_all(b"Goodbye!\n").await?;
                break;
            }
            ["screenshot"] => {
                execute_command(&command_tx, DebugCommand::Screenshot { path: None }).await
            }
            ["screenshot", path] => {
                execute_command(
                    &command_tx,
                    DebugCommand::Screenshot {
                        path: Some(PathBuf::from(path)),
                    },
                )
                .await
            }
            _ => format!("Unknown command: {}\nType 'help' for available commands\n", input),
        };

        writer.write_all(response.as_bytes()).await?;
        writer.write_all(b"> ").await?;
    }

    Ok(())
}

#[allow(dead_code)]
async fn execute_command(
    command_tx: &mpsc::Sender<(DebugCommand, mpsc::Sender<DebugResponse>)>,
    command: DebugCommand,
) -> String {
    let (response_tx, mut response_rx) = mpsc::channel(1);

    if command_tx.send((command, response_tx)).await.is_err() {
        return "Error: Renderer not responding\n".to_string();
    }

    match tokio::time::timeout(std::time::Duration::from_secs(5), response_rx.recv()).await {
        Ok(Some(DebugResponse::ScreenshotSaved { path })) => {
            format!("Screenshot saved: {}\n", path.display())
        }
        Ok(Some(DebugResponse::ScreenshotPending { path })) => {
            format!("Screenshot capture started: {}\n", path.display())
        }
        Ok(Some(DebugResponse::Error { message })) => {
            format!("Error: {}\n", message)
        }
        Ok(None) => "Error: No response from renderer\n".to_string(),
        Err(_) => "Error: Timeout waiting for response\n".to_string(),
    }
}

/// Non-blocking screenshot capture using GPU buffer copy
pub struct ScreenshotCapture {
    /// Pending captures waiting for GPU completion
    pending: Vec<PendingCapture>,
}

struct PendingCapture {
    buffer: Arc<wgpu::Buffer>,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    path: PathBuf,
    /// Receiver for map completion (Some = map started, None = not started yet)
    map_receiver: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl ScreenshotCapture {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Check if there are pending captures waiting for GPU completion
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Start a non-blocking screenshot capture
    /// Returns immediately after submitting the copy command
    pub fn start_capture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
        path: PathBuf,
    ) -> Result<PathBuf, String> {
        // Calculate padded row size (wgpu requires 256-byte alignment)
        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) / align * align;
        let buffer_size = (padded_bytes_per_row * height) as u64;

        // Create staging buffer
        let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Screenshot Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        // Copy texture to buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Screenshot Encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Store pending capture info (map will be started on first poll)
        self.pending.push(PendingCapture {
            buffer,
            width,
            height,
            padded_bytes_per_row,
            path: path.clone(),
            map_receiver: None,
        });

        Ok(path)
    }

    /// Poll pending captures and finalize any that are ready
    /// Call this each frame to check for completed captures
    pub fn poll_pending(&mut self, device: &wgpu::Device) {
        // Poll GPU once
        let _ = device.poll(wgpu::PollType::Poll);

        // Process all pending captures
        let pending = std::mem::take(&mut self.pending);

        for mut capture in pending {
            // Start map if not already started
            let rx = if let Some(rx) = capture.map_receiver.take() {
                rx
            } else {
                let buffer = capture.buffer.clone();
                let buffer_slice = buffer.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });
                rx
            };

            // Check if ready
            match rx.try_recv() {
                Ok(Ok(())) => {
                    // Buffer is mapped, save to file
                    Self::save_buffer_to_png(
                        &capture.buffer,
                        capture.width,
                        capture.height,
                        capture.padded_bytes_per_row,
                        &capture.path,
                    );
                }
                Ok(Err(e)) => {
                    tracing::error!("Screenshot buffer map failed: {:?}", e);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Not ready yet, save receiver and put capture back
                    capture.map_receiver = Some(rx);
                    self.pending.push(capture);
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    tracing::error!("Screenshot channel disconnected");
                }
            }
        }
    }

    fn save_buffer_to_png(
        buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
        padded_bytes_per_row: u32,
        path: &PathBuf,
    ) {
        let buffer_slice = buffer.slice(..);
        let data = buffer_slice.get_mapped_range();

        // Remove row padding and convert BGRA to RGBA
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            let row_start = (y * padded_bytes_per_row) as usize;
            let row_end = row_start + (width * 4) as usize;
            let row = &data[row_start..row_end];

            // Convert BGRA to RGBA
            for chunk in row.chunks(4) {
                pixels.push(chunk[2]); // R (was B)
                pixels.push(chunk[1]); // G
                pixels.push(chunk[0]); // B (was R)
                pixels.push(chunk[3]); // A
            }
        }

        drop(data);
        buffer.unmap();

        // Encode as PNG in background thread
        let path = path.clone();
        std::thread::spawn(move || {
            if let Err(e) = save_png(&path, width, height, &pixels) {
                tracing::error!("Failed to save screenshot: {}", e);
            } else {
                tracing::info!("Screenshot saved: {}", path.display());
            }
        });
    }
}

fn save_png(path: &PathBuf, width: u32, height: u32, pixels: &[u8]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let writer = std::io::BufWriter::new(file);

    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("Failed to write PNG header: {}", e))?;

    writer
        .write_image_data(pixels)
        .map_err(|e| format!("Failed to write PNG data: {}", e))?;

    Ok(())
}

impl Default for ScreenshotCapture {
    fn default() -> Self {
        Self::new()
    }
}
