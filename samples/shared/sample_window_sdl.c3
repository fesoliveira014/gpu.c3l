module sample_window;

import sdl;
import gpu;

faultdef WINDOW_INIT_FAILED, UNSUPPORTED_VIDEO_DRIVER;

struct SampleWindow {
    sdl::Window* window;
}

struct WindowEvents {
    bool quit;
    bool resized;
}

fn SampleWindow? create_sample_window(ZString title, uint width, uint height) {
    if (!sdl::init({ .video })) return WINDOW_INIT_FAILED~;
    sdl::Window* window = sdl::create_window((char*)title, (int)width, (int)height, { .resizable });
    if (window == null) {
        sdl::quit();
        return WINDOW_INIT_FAILED~;
    }
    return (SampleWindow){ .window = window };
}

fn void destroy_sample_window(SampleWindow* sample) {
    if (sample.window != null) {
        sdl::destroy_window(sample.window);
        sample.window = null;
    }
    sdl::quit();
}

<* Extract platform-native handles from the SDL window into the gpu-neutral
   SurfaceDesc, branching on SDL's active video driver. *>
fn gpu::SurfaceDesc? surface_desc_for(SampleWindow* sample) {
    sdl::PropertiesID props = sdl::get_window_properties(sample.window);
    ZString driver = (ZString)sdl::get_current_video_driver();

    if (driver.str_view() == "wayland") {
        return (gpu::SurfaceDesc){
            .platform       = gpu::PlatformKind.WAYLAND,
            .native_display = sdl::get_pointer_property(props, (char*)sdl::WindowProperties.WAYLAND_DISPLAY_POINTER, null),
            .native_window  = sdl::get_pointer_property(props, (char*)sdl::WindowProperties.WAYLAND_SURFACE_POINTER, null),
        };
    }
    if (driver.str_view() == "x11") {
        return (gpu::SurfaceDesc){
            .platform       = gpu::PlatformKind.X11,
            .native_display = sdl::get_pointer_property(props, (char*)sdl::WindowProperties.X11_DISPLAY_POINTER, null),
            .native_window  = (void*)(uptr)sdl::get_number_property(props, (char*)sdl::WindowProperties.X11_WINDOW_NUMBER, 0),
        };
    }
    if (driver.str_view() == "windows") {
        return (gpu::SurfaceDesc){
            .platform       = gpu::PlatformKind.WIN32,
            .native_display = sdl::get_pointer_property(props, (char*)sdl::WindowProperties.WIN32_INSTANCE_POINTER, null),
            .native_window  = sdl::get_pointer_property(props, (char*)sdl::WindowProperties.WIN32_HWND_POINTER, null),
        };
    }
    return UNSUPPORTED_VIDEO_DRIVER~;
}

fn void pixel_size(SampleWindow* sample, uint* width, uint* height) {
    int w = 0;
    int h = 0;
    sdl::get_window_size_in_pixels(sample.window, &w, &h);
    *width = w < 0 ? 0 : (uint)w;
    *height = h < 0 ? 0 : (uint)h;
}

fn WindowEvents pump_events(SampleWindow* sample) {
    WindowEvents events;
    sdl::Event event;
    while (sdl::poll_event(&event)) {
        switch (event.type) {
            case sdl::EventType.QUIT:
                events.quit = true;
            case sdl::EventType.WINDOW_RESIZED:
            case sdl::EventType.WINDOW_PIXEL_SIZE_CHANGED:
                events.resized = true;
            default:
                break;
        }
    }
    return events;
}
