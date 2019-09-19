pub mod windows {
    use std::io;
    use std::path::PathBuf;

    /// Returns path to `{user}/AppData/Local`.
    ///
    /// It needs to be retrieved using system calls as %localappdata% might
    /// not be set in some circumstances.
    #[allow(dead_code)]
    pub fn localappdata_path() -> io::Result<PathBuf> {
        use std::ffi::OsString;
        use std::os::windows::ffi::OsStringExt;
        use std::ptr;
        use std::slice;

        let mut path = ptr::null_mut();
        let result;

        unsafe {
            let code = winapi::um::shlobj::SHGetKnownFolderPath(
                &winapi::um::knownfolders::FOLDERID_LocalAppData,
                0,
                ptr::null_mut(),
                &mut path,
            );

            if code == 0 {
                let mut length = 0usize;

                while *path.add(length) != 0 {
                    length += 1;
                }

                let slice = slice::from_raw_parts(path, length);
                result = Ok(OsString::from_wide(slice).into());
            } else {
                result = Err(io::Error::from_raw_os_error(code));
            }
            winapi::um::combaseapi::CoTaskMemFree(path as *mut _);
        }

        result
    }
}
