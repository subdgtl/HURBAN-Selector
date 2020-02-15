use std::borrow::Cow;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationLevel {
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notification {
    live_until: Instant,
    pub level: NotificationLevel,
    pub text: Cow<'static, str>,
}

pub struct Notifications {
    ttl: Duration,
    notifications: VecDeque<Notification>,
}

impl Notifications {
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            ttl,
            notifications: VecDeque::new(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Notification> {
        self.notifications.iter()
    }

    pub fn update(&mut self, time: Instant) {
        while let Some(notification) = self.notifications.front() {
            if time < notification.live_until {
                break;
            }

            self.notifications.pop_front();
        }
    }

    pub fn push<S: Into<Cow<'static, str>>>(
        &mut self,
        time: Instant,
        level: NotificationLevel,
        text: S,
    ) {
        self.notifications.push_back(Notification {
            live_until: time + self.ttl,
            level,
            text: text.into(),
        });
    }
}
