# Windows installer

A Windows installer based on [Wix](https://wixtoolset.org).

## Creating the installer

Make sure you have Wix installed. Get it at their [github releases
page](https://github.com/wixtoolset/wix3/releases/tag/wix3112rtm). Wix contains
two important tools: `candle.exe` and `light.exe` in the `bin` folder of their
installation directory, e.g. `/c/Program Files (x86)/WiX Toolset
v3.11/bin/candle.exe`. Note that the tools might not be in your PATH.

To create a distribution, run the following in the project directory:

- `cargo build --release --features dist`
- `candle.exe 'installer_windows/H.U.R.B.A.N. selector.wxs'`. This will
  output a `H.U.R.B.A.N. selector.wixobj` file in the current directory.
- `light.exe -ext WixUIExtension 'H.U.R.B.A.N. selector.wixobj'`. This will
  output a `H.U.R.B.A.N. selector.msi` installer in the current directory.

Note: running `light.exe` will produce some warnings. This is ok and is caused
by the way the Windows C runtime library we redistribute is authored.

## Modifying the installer

Some documentation and resource links are included in
`installer_windows/H.U.R.B.A.N. selector.wxs`. Good luck!
