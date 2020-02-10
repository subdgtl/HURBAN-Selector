# Windows installer

An Inno Setup Windows installer.

## Creating the installer

Make sure you have the following installed:

- [Inno Setup](http://www.jrsoftware.org/isinfo.php),
- [rcedit](https://github.com/electron/rcedit) (binary releases can be found
  [the github releases page](https://github.com/electron/rcedit/releases)).

To create a distribution follow these steps:

- `cargo build --release --features dist`,
- Add icon to binary:
 `rcedit-x86.exe "C:\path\to\release\hurban_selector.exe" --set-icon "C:\path\to\icons\64x64.ico"`,
- Compile `installer/setup.iss` from Inno Setup IDE,
- Output file will be located at `installer/bin`.

Note that window icon and desktop / Start menu icons differ. Window icon is
taken from `icons/64x64.ico` and baked into application during compilation.
Desktop icon is set with `rcedit` and can be an entirely different file.

Installer itself will also contain [Microsoft Visual C++ Redistributable for
Visual Studio 2015, 2017 and 2019](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)
that is needed to run the application. Code used to install this dependency is
a modified version of [Inno Dependency Installer](https://github.com/domgho/innodependencyinstaller).

The only supported version is 64-bit and only English language is available.
