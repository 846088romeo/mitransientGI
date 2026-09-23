$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($userPath -notlike "*C:\ffmpeg\bin*") {
    [Environment]::SetEnvironmentVariable(
        "Path",
        "$userPath;C:\ffmpeg\bin",
        "User"
    )
}
