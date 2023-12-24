use crate::DynResult;
use kira::{
    manager::{backend::DefaultBackend, AudioManager, AudioManagerSettings},
    sound::static_sound::{StaticSoundData, StaticSoundSettings},
};
use lazy_static::lazy_static;
use std::{
    io::{Cursor, Read, Seek},
    sync::{Arc, Mutex},
};
use symphonia_core::io::MediaSource;

static RAW_AUDIO: &[u8] = include_bytes!("../audio/bad-thoughts-remastered.ogg");

lazy_static! {
    static ref AUDIO_MANAGER: Arc<Mutex<AudioManager<DefaultBackend>>> = Arc::new(Mutex::new(
        AudioManager::<DefaultBackend>::new(AudioManagerSettings::default()).unwrap()
    ));
}

pub fn play_a_song() -> DynResult<()> {
    let audio_source = RawAudioSource::new(RAW_AUDIO);
    let settings = StaticSoundSettings::default().volume(0.3);
    let sound_data = StaticSoundData::from_media_source(audio_source, settings)?;

    AUDIO_MANAGER.lock()?.play(sound_data)?;

    Ok(())
}

pub struct RawAudioSource {
    len: u64,
    data: Cursor<Vec<u8>>,
}

impl RawAudioSource {
    pub fn new(data: &[u8]) -> Self {
        Self {
            len: data.len() as u64,
            data: Cursor::new(data.to_vec()),
        }
    }
}

impl MediaSource for RawAudioSource {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        Some(self.len)
    }
}

impl Seek for RawAudioSource {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.data.seek(pos)
    }
}

impl Read for RawAudioSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.data.read(buf)
    }
}
