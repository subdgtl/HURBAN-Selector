const DEFAULT_APP_LOG_LEVEL: LogLevel = LogLevel::Debug;
const DEFAULT_LIB_LOG_LEVEL: LogLevel = LogLevel::Warning;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Off,
    Error,
    Warning,
    Info,
    Debug,
    Trace,
}

impl Into<log::LevelFilter> for LogLevel {
    fn into(self) -> log::LevelFilter {
        match self {
            Self::Off => log::LevelFilter::Off,
            Self::Error => log::LevelFilter::Error,
            Self::Warning => log::LevelFilter::Warn,
            Self::Info => log::LevelFilter::Info,
            Self::Debug => log::LevelFilter::Debug,
            Self::Trace => log::LevelFilter::Trace,
        }
    }
}

/// Initializes logger for current environment.
///
/// In case of any error, basic logger that does nothing, is returned. Errors
/// can possibly happen in case of filesystem permission errors in dist build.
pub fn init(app_log_level: Option<LogLevel>, lib_log_level: Option<LogLevel>) {
    let base_logger = fern::Dispatch::new();
    let app_level_filter: log::LevelFilter = app_log_level.unwrap_or(DEFAULT_APP_LOG_LEVEL).into();
    let lib_level_filter: log::LevelFilter = lib_log_level.unwrap_or(DEFAULT_LIB_LOG_LEVEL).into();

    let env_specific_logger = init_env_specific(base_logger, app_level_filter, lib_level_filter);

    env_specific_logger.apply().expect("Failed to build logger");
}

#[cfg(not(feature = "dist"))]
pub fn init_env_specific(
    base_logger: fern::Dispatch,
    app_level_filter: log::LevelFilter,
    lib_level_filter: log::LevelFilter,
) -> fern::Dispatch {
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

    base_logger
        .level(lib_level_filter)
        .level_for("hurban_selector", app_level_filter)
        .chain(std::io::stdout())
}

#[cfg(feature = "dist")]
pub fn init_env_specific(
    base_logger: fern::Dispatch,
    app_level_filter: log::LevelFilter,
    lib_level_filter: log::LevelFilter,
) -> fern::Dispatch {
    use std::fs;
    use std::path::Path;

    #[cfg(target_os = "windows")]
    let path = match dirs::data_local_dir() {
        Some(appdata_local) => Path::new(&appdata_local).join("HURBAN Selector/Logs"),
        None => return base_logger,
    };

    #[cfg(target_os = "macos")]
    let path = {
        use std::env;

        match env::var("HOME") {
            Ok(home_dir) => Path::new(&home_dir).join("Library/Logs/HURBAN_Selector"),
            Err(_) => return base_logger,
        }
    };

    #[cfg(target_os = "linux")]
    let path = Path::new("/var/log/HURBAN_Selector").to_path_buf();

    if !path.exists() {
        let result = fs::create_dir_all(&path);

        if result.is_err() {
            return base_logger;
        }
    }

    let today_format = chrono::Local::today().format("%Y-%m-%d");
    let file_name = path.join(format!("{}.log", today_format));
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
        .level(lib_level_filter)
        .level_for("hurban_selector", app_level_filter)
        .chain(file)
}
