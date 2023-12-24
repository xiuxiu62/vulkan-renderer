mod audio;
mod graphics;
mod shaders;

type DynResult<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> DynResult<()> {
    std::thread::spawn(|| {
        audio::play_a_song().unwrap();
    });

    graphics::Runtime::new()?.run()
}
