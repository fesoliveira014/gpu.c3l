module gpu::internal::vk @private @feat(!WIN32);

import std::thread;

alias RecordingThreadIdentity = Thread;

fn RecordingThreadIdentity current_recording_thread_identity() =>
    thread::current();

fn bool recording_thread_identity_matches(RecordingThreadIdentity identity) =>
    identity.equals(thread::current());
