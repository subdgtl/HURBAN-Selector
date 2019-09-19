use std::env;

const LIBS_DEFAULT_LOG_LEVEL: log::LevelFilter = log::LevelFilter::Warn;

/// Initializes logger for current environment.
///
/// In case of any error, basic logger, that does nothing, is returned. Errors
/// can possibly happen in case of file permission errors in release build.
pub fn init() {
    let base_logger = fern::Dispatch::new();
    let env_specific_logger = init_env_specific(base_logger);

    env_specific_logger.apply().expect("Failed to build logger");
}

#[cfg(debug_assertions)]
pub fn init_env_specific(base_logger: fern::Dispatch) -> fern::Dispatch {
    use fern::colors::{Color, ColoredLevelConfig};

    let colors = ColoredLevelConfig::new()
        .error(Color::Red)
        .warn(Color::Yellow)
        .info(Color::Cyan)
        .debug(Color::BrightWhite)
        .trace(Color::White);
    let base_logger = base_logger.format(move |out, message, record| {
        out.finish(format_args!(
            "{} [{}] [{}] {}",
            chrono::Local::now().format("[%Y-%m-%d %H:%M:%S]"),
            record.target(),
            colors.color(record.level()),
            message
        ))
    });
    let libs_log_level = env::var("HS_LIBS_LOG_LEVEL")
        .map(|libs_log_level| match libs_log_level.as_str() {
            "error" => log::LevelFilter::Error,
            "warning" => log::LevelFilter::Warn,
            "info" => log::LevelFilter::Info,
            "debug" => log::LevelFilter::Debug,
            "trace" => log::LevelFilter::Trace,
            "off" => log::LevelFilter::Off,
            _ => LIBS_DEFAULT_LOG_LEVEL,
        })
        .unwrap_or(LIBS_DEFAULT_LOG_LEVEL);

    base_logger
        .level(libs_log_level)
        .level_for("hurban_selector", log::LevelFilter::Debug)
        .chain(std::io::stdout())
}

#[cfg(not(debug_assertions))]
pub fn init_env_specific(base_logger: fern::Dispatch) -> fern::Dispatch {
    use std::fs;
    use std::path::Path;

    let path = if cfg!(target_os = "windows") {
        let appdata = env::var("localappdata").expect("%localappdata% should be defined in env");

        Ok(Path::new(&appdata).join("HURBAN Selector/Logs"))
    } else if cfg!(target_os = "macos") {
        let home_dir = env::var("HOME").expect("$HOME should be defined in env");

        Ok(Path::new(&home_dir).join("Library/Logs/HURBAN_Selector"))
    } else if cfg!(target_os = "linux") {
        Ok(Path::new("/var/log/HURBAN_Selector").to_path_buf())
    } else {
        Err(())
    };

    let path_buf = match path {
        Ok(path_buf) => path_buf,
        Err(_) => return base_logger,
    };

    if !path_buf.exists() {
        let result = fs::create_dir_all(&path_buf);

        if result.is_err() {
            return base_logger;
        }
    }

    let today_format = chrono::Local::today().format("%Y-%m-%d");
    let file_name = path_buf.join(format!("{}.log", today_format));
    let file = match fern::log_file(file_name) {
        Ok(file) => file,
        Err(_) => return base_logger,
    };

    base_logger
        .format(move |out, message, record| {
            out.finish(format_args!(
                "{} [{}] [{}] {}",
                chrono::Local::now().format("[%Y-%m-%d %H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level(LIBS_DEFAULT_LOG_LEVEL)
        .level_for("hurban_selector", log::LevelFilter::Info)
        .chain(file)
}
