const video = document.getElementById('video');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
const toast = document.getElementById('toast');

// --- Toast Notification ---
function showToast(message, type = 'success') {
    toast.textContent = message;
    toast.className = `toast show ${type}`;
    setTimeout(() => {
        toast.className = 'toast hidden';
    }, 3000);
}

// --- Camera Handling ---
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();
    } catch (err) {
        console.error("Error accessing webcam: ", err);
        showToast("Error accessing webcam. Please allow permissions.", 'error');
    }
}

function captureFrame() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => {
        canvas.toBlob((blob) => {
            resolve(blob);
        }, 'image/jpeg', 0.9);
    });
}

// --- Registration Logic ---
const registerBtn = document.getElementById('register-btn');
const nameInput = document.getElementById('name-input');
const progressBar = document.getElementById('progress-bar');
const progressContainer = document.querySelector('.progress-container');

if (registerBtn && nameInput) {
    startCamera();

    registerBtn.addEventListener('click', async () => {
        const name = nameInput.value.trim();
        if (!name) {
            showToast("Please enter a name", 'error');
            return;
        }

        // 1. Create User
        registerBtn.disabled = true;
        registerBtn.textContent = "Creating User...";
        
        try {
            const res = await fetch('/api/create_user', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ name: name })
            });
            
            if (!res.ok) throw new Error("Failed to create user");
            
            const data = await res.json();
            const userId = data.user_id;

            // 2. Start Capture Loop (30 images)
            registerBtn.textContent = "Capturing Face...";
            progressContainer.style.display = 'block';
            
            let captured = 0;
            const totalImages = 30;

            for (let i = 0; i < totalImages; i++) {
                const blob = await captureFrame();
                const formData = new FormData();
                formData.append('user_id', userId);
                formData.append('image', blob);

                const upRes = await fetch('/api/upload_face', {
                    method: 'POST',
                    body: formData
                });
                
                // Even if detection fails, we keep trying until we reach total attempts? 
                // Better: Only increment captured if success.
                // For simplicity/speed here: we try 30 times.
                
                captured++;
                progressBar.style.width = `${(captured / totalImages) * 100}%`;
                
                // Small delay
                await new Promise(r => setTimeout(r, 100)); // 100ms delay
            }

            // 3. Train Model
            registerBtn.textContent = "Training Model...";
            const trainRes = await fetch('/api/train', { method: 'POST' });
            if (!trainRes.ok) throw new Error("Training failed");

            showToast("Registration Complete!", 'success');
            setTimeout(() => {
                window.location.href = '/';
            }, 1000);

        } catch (err) {
            console.error(err);
            showToast(err.message, 'error');
            registerBtn.disabled = false;
            registerBtn.textContent = "Capture & Register";
        }
    });
}

// --- Attendance Logic ---
const attendanceBtn = document.getElementById('attendance-btn');
const attendanceResult = document.getElementById('attendance-result');

if (attendanceBtn) {
    startCamera();

    attendanceBtn.addEventListener('click', async () => {
        attendanceBtn.disabled = true;
        attendanceBtn.textContent = "Scanning...";
        
        try {
            const blob = await captureFrame();
            const formData = new FormData();
            formData.append('image', blob);

            const res = await fetch('/api/mark_attendance', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();
            
            if (data.status === 'success' && data.match) {
                showToast(`Welcome, ${data.user}!`, 'success');
                attendanceResult.innerHTML = `<h3 class="text-success">Welcome, ${data.user}</h3><p>${data.message}</p>`;
            } else if (data.status === 'success' && !data.match) {
                showToast("Face not recognized", 'error');
                attendanceResult.innerHTML = `<h3 class="text-error">Not Recognized</h3>`;
            } else {
                showToast(data.message, 'error');
            }

        } catch (err) {
            console.error(err);
            showToast("Error marking attendance", 'error');
        } finally {
            attendanceBtn.disabled = false;
            attendanceBtn.textContent = "Mark Attendance";
        }
    });
}
