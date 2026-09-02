module gpu::internal::vk @private @feat(WIN32);

import std::os::win32;

alias RecordingThreadIdentity = Win32_DWORD;

fn RecordingThreadIdentity current_recording_thread_identity() =>
    win32::getCurrentThreadId();

fn bool recording_thread_identity_matches(RecordingThreadIdentity identity) =>
    identity == win32::getCurrentThreadId();
