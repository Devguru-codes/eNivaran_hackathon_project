document.addEventListener('DOMContentLoaded', function () {
    // Pothole stats fetch and render
    function fetchPotholeStats() {
        fetch('/pothole_stats')
            .then(res => res.json())
            .then(data => {
                document.getElementById('stat-total').textContent = data.total_potholes ?? '-';
                document.getElementById('stat-high').textContent = data.high_priority_count ?? '-';
                document.getElementById('stat-medium').textContent = data.medium_priority_count ?? '-';
                document.getElementById('stat-low').textContent = data.low_priority_count ?? '-';
                document.getElementById('stat-updated').textContent = data.last_updated
                    ? new Date(data.last_updated).toLocaleString()
                    : '-';
            })
            .catch(() => {
                document.getElementById('stat-total').textContent = '-';
                document.getElementById('stat-high').textContent = '-';
                document.getElementById('stat-medium').textContent = '-';
                document.getElementById('stat-low').textContent = '-';
                document.getElementById('stat-updated').textContent = '-';
            });
    }
    fetchPotholeStats();

    const form = document.getElementById('pothole-form');
    const resultSection = document.getElementById('result-section');
    const resultJson = document.getElementById('result-json');
    const annotatedImage = document.getElementById('annotated-image');
    const errorSection = document.getElementById('error-section');

    form.addEventListener('submit', function (e) {
        e.preventDefault();
        resultSection.style.display = 'none';
        errorSection.style.display = 'none';
        resultJson.textContent = '';
        annotatedImage.src = '';

        // Show loading spinner and disable button
        const loading = document.getElementById('pothole-loading');
        const submitBtn = form.querySelector('button[type="submit"]');
        loading.style.display = 'flex';
        submitBtn.disabled = true;

        const formData = new FormData(form);

        fetch('/detect_pothole', {
            method: 'POST',
            body: formData
        })
        .then(async response => {
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Detection failed');
            }
            return response.json();
        })
        .then(data => {
            resultJson.textContent = JSON.stringify(data.result, null, 2);
            annotatedImage.src = 'data:image/jpeg;base64,' + data.annotated_image_b64;
            resultSection.style.display = 'block';
            fetchPotholeStats(); // Refresh stats after detection
        })
        .catch(err => {
            errorSection.textContent = err.message;
            errorSection.style.display = 'block';
        })
        .finally(() => {
            loading.style.display = 'none';
            submitBtn.disabled = false;
        });
    });

    // Complaint form logic
    const complaintForm = document.getElementById('complaint-form');
    const complaintResultSection = document.getElementById('complaint-result-section');
    const complaintResultMessage = document.getElementById('complaint-result-message');
    const complaintErrorSection = document.getElementById('complaint-error-section');

    if (complaintForm) {
        complaintForm.addEventListener('submit', function (e) {
            e.preventDefault();
            complaintResultSection.style.display = 'none';
            complaintErrorSection.style.display = 'none';
            complaintResultMessage.textContent = '';

            // Show loading spinner and disable button
            const loading = document.getElementById('complaint-loading');
            const submitBtn = complaintForm.querySelector('button[type="submit"]');
            loading.style.display = 'flex';
            submitBtn.disabled = true;

            const formData = new FormData(complaintForm);

            fetch('/raise_complaint', {
                method: 'POST',
                body: formData
            })
            .then(async response => {
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Complaint submission failed');
                }
                return response.json();
            })
            .then(data => {
                complaintResultMessage.textContent = data.message || 'Complaint submitted successfully.';
                complaintResultSection.style.display = 'block';
            })
            .catch(err => {
                complaintErrorSection.textContent = err.message;
                complaintErrorSection.style.display = 'block';
            })
            .finally(() => {
                loading.style.display = 'none';
                submitBtn.disabled = false;
            });
        });
    }
});
