#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::Clap)]
pub enum LogLevel {
    Off,
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl Into<log::LevelFilter> for LogLevel {
    fn into(self) -> log::LevelFilter {
        match self {
            Self::Off => log::LevelFilter::Off,
            Self::Error => log::LevelFilter::Error,
            Self::Warn => log::LevelFilter::Warn,
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
pub fn init(log_level_app: LogLevel, log_level_lib: LogLevel) {
    let base_logger = fern::Dispatch::new();
    let app_level_filter: log::LevelFilter = log_level_app.into();
    let lib_level_filter: log::LevelFilter = log_level_lib.into();

    let env_specific_logger = init_env_specific(base_logger, app_level_filter, lib_level_filter);

    env_specific_logger.apply().expect("Failed to build logger");
}

#[cfg(not(feature = "dist"))]
pub fn init_env_specific(
    base_logger: fern::Dispatch,
    filter_app: log::LevelFilter,
    filter_lib: log::LevelFilter,
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
        .level(filter_lib)
        .level_for("hurban_selector", filter_app)
        .chain(std::io::stdout())
}

#[cfg(feature = "dist")]
pub fn init_env_specific(
    base_logger: fern::Dispatch,
    filter_app: log::LevelFilter,
    filter_lib: log::LevelFilter,
) -> fern::Dispatch {
    use std::fs;
    use std::path::Path;

    #[cfg(target_os = "windows")]
    let path = match dirs::data_local_dir() {
        Some(appdata_local) => Path::new(&appdata_local).join("H.U.R.B.A.N. selector/Logs"),
        None => return base_logger,
    };

    #[cfg(target_os = "macos")]
    let path = {
        use std::env;

        match env::var("HOME") {
            Ok(home_dir) => Path::new(&home_dir).join("Library/Logs/H.U.R.B.A.N. selector"),
            Err(_) => return base_logger,
        }
    };

    #[cfg(target_os = "linux")]
    let path = Path::new("/var/log/H.U.R.B.A.N. selector").to_path_buf();

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
        .level(filter_lib)
        .level_for("hurban_selector", filter_app)
        .chain(file)
}
