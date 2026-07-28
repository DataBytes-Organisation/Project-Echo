const DEFAULT_AUDIO_PATH = './audio/default.mp3';
const ALLOWED_EXTENSIONS = ['.mp3', '.wav', '.ogg'];

export function sanitisePath(path) {
    if (typeof path !== 'string') return null;
    
    const sanitised = path.replace(/[^a-zA-Z0-9\-_./]/g, '');
    
    if (sanitised.includes('../')) return null;
    
    const hasValidExtension = ALLOWED_EXTENSIONS.some(ext => sanitised.endsWith(ext));
    if (!hasValidExtension) return null;
    
    return sanitised;
}

export const animalAudio = new Audio(DEFAULT_AUDIO_PATH);

export function setAnimalAudio(rawPath) {
    const safePath = sanitisePath(rawPath);
    
    if (!safePath) {
        console.warn('Invalid audio path, falling back to default.');
        animalAudio.src = DEFAULT_AUDIO_PATH;
    } else {
        animalAudio.src = safePath;
    }
    
    animalAudio.load();
    return animalAudio.src;
}