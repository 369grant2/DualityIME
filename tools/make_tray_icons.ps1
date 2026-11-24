# Draws the three icons the input method shows on the taskbar (cpp\tip\*.ico), the way Windows 11
# shows its own input methods there: the bare character in white, with a black rim so the same
# icon reads on a light taskbar too.
#   icon_profile.ico  the input method's own: U+5100 (yi)
#   icon.ico          the mode button in mixed mode: U+5169 (liang)
#   icon_en.ico       the mode button in English mode: U+82F1 (ying)
# The drawing is in tools\tray_icons.cs. Needs the Microsoft JhengHei font (part of Windows).
#   powershell -ExecutionPolicy Bypass -File tools\make_tray_icons.ps1
# (installer\app.ico and settings\ui\logo.svg still come from tools\make_icons.py.)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$tip = Join-Path $root "cpp\tip"
Add-Type -AssemblyName System.Drawing
Add-Type -Path (Join-Path $PSScriptRoot "tray_icons.cs") -ReferencedAssemblies System.Drawing

$sizes = 16, 20, 24, 32, 40, 48, 64, 256
# (code points, not the characters: Windows PowerShell reads a script without a BOM as ANSI)
$icons = [ordered]@{
    "icon_profile.ico" = [string][char]0x5100
    "icon.ico"         = [string][char]0x5169
    "icon_en.ico"      = [string][char]0x82F1
}
foreach ($name in $icons.Keys) {
    $frames = New-Object 'System.Collections.Generic.List[System.Drawing.Bitmap]'
    foreach ($size in $sizes) { $frames.Add([IconDraw]::Draw($icons[$name], $size)) }
    [IconDraw]::SaveIco((Join-Path $tip $name), $frames)
    Write-Host "wrote cpp\tip\$name"
}
