document.addEventListener("DOMContentLoaded", () => {
    const queryForm = document.getElementById("queryForm");
    const queryInput = document.getElementById("queryInput");
    const submitBtn = document.getElementById("submitBtn");
    const loadingState = document.getElementById("loadingState");
    const resultContainer = document.getElementById("resultContainer");

    // Telemetry Elements
    const intentBadge = document.getElementById("intentBadge");
    const gaugeBar = document.getElementById("gaugeBar");
    const gaugeValue = document.getElementById("gaugeValue");
    const retryCount = document.getElementById("retryCount");
    const healingStatus = document.getElementById("healingStatus");
    const failureType = document.getElementById("failureType");
    const latencyValue = document.getElementById("latencyValue");
    const globalTotalQueries = document.getElementById("globalTotalQueries");
    const globalPassRate = document.getElementById("globalPassRate");

    // Output Elements
    const answerText = document.getElementById("answerText");
    const groundedBadge = document.getElementById("groundedBadge");
    const evidenceList = document.getElementById("evidenceList");
    const healingHistory = document.getElementById("healingHistory");
    const telemetryDetail = document.getElementById("telemetryDetail");

    // Session Elements
    const sessionsList = document.getElementById("sessionsList");
    const newSessionBtn = document.getElementById("newSessionBtn");

    // Auth Elements
    const userDisplay = document.getElementById("userDisplay");
    const authModalBtn = document.getElementById("authModalBtn");
    const authModal = document.getElementById("authModal");
    const closeAuthModal = document.getElementById("closeAuthModal");
    const authForm = document.getElementById("authForm");
    const authTitle = document.getElementById("authTitle");
    const authUsername = document.getElementById("authUsername");
    const authPassword = document.getElementById("authPassword");
    const authError = document.getElementById("authError");
    const authSubmitBtn = document.getElementById("authSubmitBtn");
    const toggleAuthModeBtn = document.getElementById("toggleAuthModeBtn");

    // Upload & Drag-and-Drop Elements
    const dropZone = document.getElementById("dropZone");
    const uploadForm = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");
    const fileNameDisplay = document.getElementById("fileNameDisplay");
    const uploadBtn = document.getElementById("uploadBtn");
    const uploadStatus = document.getElementById("uploadStatus");

    let selectedFile = null;

    function handleFileSelection(file) {
        if (!file) return;
        selectedFile = file;
        fileNameDisplay.textContent = file.name;
        uploadBtn.disabled = false;
    }

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelection(e.target.files[0]);
            }
        });
    }

    // Drag and Drop Handling
    if (dropZone) {
        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add("drag-over");
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove("drag-over");
            }, false);
        });

        dropZone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length > 0) {
                handleFileSelection(dt.files[0]);
            }
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const fileToUpload = selectedFile || (fileInput.files && fileInput.files[0]);
            if (!fileToUpload) return;

            const formData = new FormData();
            formData.append("file", fileToUpload);

            uploadBtn.disabled = true;
            uploadStatus.classList.remove("hidden");
            uploadStatus.textContent = `Uploading & processing '${fileToUpload.name}' into vector database...`;
            uploadStatus.style.borderColor = "var(--primary)";
            uploadStatus.style.color = "var(--accent)";

            try {
                const response = await fetch("/api/upload", {
                    method: "POST",
                    body: formData,
                });

                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || "Upload failed.");
                }

                uploadStatus.textContent = `✓ ${data.message} (${data.details.num_chunks} chunks indexed)`;
                uploadStatus.style.borderColor = "var(--success)";
                uploadStatus.style.color = "var(--success)";
                fileInput.value = "";
                selectedFile = null;
                fileNameDisplay.textContent = "No file selected";
            } catch (err) {
                uploadStatus.textContent = `✗ Upload Error: ${err.message}`;
                uploadStatus.style.borderColor = "var(--danger)";
                uploadStatus.style.color = "var(--danger)";
            } finally {
                uploadBtn.disabled = true;
            }
        });
    }

    let isRegisterMode = false;
    let storedUser = localStorage.getItem("selfrag_user");
    let currentUser = storedUser ? JSON.parse(storedUser) : null;
    let currentSessionId = "sess_" + Math.random().toString(36).substring(2, 10);

    function requireLogin() {
        if (!currentUser) {
            authModal.classList.remove("hidden");
            closeAuthModal.style.display = "none"; // Hide close button
            document.querySelector(".app-container").style.filter = "blur(4px)";
            document.querySelector(".app-container").style.pointerEvents = "none";
            return true;
        }
        return false;
    }

    if (!requireLogin()) {
        updateUserUI();
        fetchSessions();
        fetchTelemetry();
    }

    // Modal Control Events
    authModalBtn.addEventListener("click", () => {
        authModal.classList.remove("hidden");
        closeAuthModal.style.display = "block"; // Show close button if opened manually
    });

    closeAuthModal.addEventListener("click", () => {
        if (currentUser) {
            authModal.classList.add("hidden");
            authError.classList.add("hidden");
        }
    });

    toggleAuthModeBtn.addEventListener("click", () => {
        isRegisterMode = !isRegisterMode;
        if (isRegisterMode) {
            authTitle.textContent = "Register New Account";
            authSubmitBtn.textContent = "Create Account";
            toggleAuthModeBtn.textContent = "Switch to Sign In";
        } else {
            authTitle.textContent = "Sign In to Self-Healing RAG";
            authSubmitBtn.textContent = "Sign In";
            toggleAuthModeBtn.textContent = "Switch to Register";
        }
        authError.classList.add("hidden");
    });

    // Auth Submit
    authForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const username = authUsername.value.trim();
        const password = authPassword.value.trim();
        if (!username || !password) return;

        const endpoint = isRegisterMode ? "/api/auth/register" : "/api/auth/login";
        authSubmitBtn.disabled = true;

        try {
            const res = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, password })
            });

            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || "Authentication failed.");
            }

            currentUser = {
                user_id: data.user_id,
                username: data.username,
                api_key: data.api_key
            };

            localStorage.setItem("selfrag_user", JSON.stringify(currentUser));
            updateUserUI();
            authModal.classList.add("hidden");
            document.querySelector(".app-container").style.filter = "none";
            document.querySelector(".app-container").style.pointerEvents = "auto";
            
            authUsername.value = "";
            authPassword.value = "";

            currentSessionId = "sess_" + Math.random().toString(36).substring(2, 10);
            resultContainer.classList.add("hidden");
            fetchSessions();
            fetchTelemetry();
        } catch (err) {
            authError.textContent = err.message;
            authError.classList.remove("hidden");
        } finally {
            authSubmitBtn.disabled = false;
        }
    });

    function updateUserUI() {
        userDisplay.textContent = `User: ${currentUser.username}`;
    }

    newSessionBtn.addEventListener("click", () => {
        currentSessionId = "sess_" + Math.random().toString(36).substring(2, 10);
        queryInput.value = "";
        resultContainer.classList.add("hidden");
        fetchSessions();
    });

    // Knowledge Scope Chip Event Listeners
    let currentScope = "hybrid";
    document.querySelectorAll(".scope-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            document.querySelectorAll(".scope-chip").forEach(c => c.classList.remove("active"));
            chip.classList.add("active");
            const radio = chip.querySelector("input[type='radio']");
            if (radio) {
                radio.checked = true;
                currentScope = radio.value;
            }
        });
    });

    // Preset Chip Buttons
    document.querySelectorAll(".preset-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            queryInput.value = chip.getAttribute("data-query");
            queryForm.requestSubmit();
        });
    });

    // Tab Switching
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.add("hidden"));

            btn.classList.add("active");
            const target = btn.getAttribute("data-tab");
            document.getElementById(target).classList.remove("hidden");
        });
    });

    // Handle Query Form Submit
    queryForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        // UI Reset
        submitBtn.disabled = true;
        loadingState.classList.remove("hidden");
        resultContainer.classList.add("hidden");

        try {
            const response = await fetch("/api/query", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-User-ID": currentUser.user_id,
                    "X-Session-ID": currentSessionId,
                    "Authorization": `Bearer ${currentUser.api_key}`
                },
                body: JSON.stringify({
                    query: query,
                    session_id: currentSessionId,
                    user_id: currentUser.user_id,
                    search_scope: currentScope
                })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Query processing failed.");
            }

            const data = await response.json();
            renderResult(data);
            fetchSessions();
            fetchTelemetry();
        } catch (err) {
            alert("Error executing query: " + err.message);
        } finally {
            submitBtn.disabled = false;
            loadingState.classList.add("hidden");
        }
    });

    async function fetchSessions() {
        try {
            const res = await fetch(`/api/sessions?user_id=${currentUser.user_id}`);
            if (!res.ok) return;
            const data = await res.json();
            sessionsList.innerHTML = "";

            if (!data.sessions || data.sessions.length === 0) {
                sessionsList.innerHTML = `<div class="session-item active">New Research Thread</div>`;
                return;
            }

            data.sessions.forEach(sess => {
                const item = document.createElement("div");
                item.className = `session-item ${sess.session_id === currentSessionId ? "active" : ""}`;
                item.textContent = sess.title;
                item.addEventListener("click", () => {
                    currentSessionId = sess.session_id;
                    fetchSessionHistory(currentSessionId);
                    document.querySelectorAll(".session-item").forEach(i => i.classList.remove("active"));
                    item.classList.add("active");
                });
                sessionsList.appendChild(item);
            });
        } catch (e) {
            console.error("Failed to fetch sessions", e);
        }
    }

    async function fetchSessionHistory(sessId) {
        try {
            const res = await fetch(`/api/sessions/${sessId}`);
            if (!res.ok) return;
            const data = await res.json();
            if (data.history && data.history.length > 0) {
                const lastItem = data.history[data.history.length - 1];
                renderResult(lastItem);
            }
        } catch (e) {
            console.error("Failed to fetch session history", e);
        }
    }

    async function fetchTelemetry() {
        try {
            const res = await fetch("/api/telemetry");
            if (!res.ok) return;
            const data = await res.json();
            if (data.metrics) {
                globalTotalQueries.textContent = data.metrics.total_queries;
                globalPassRate.textContent = `${data.metrics.grounding_pass_rate}%`;

                // Render Telemetry detail tab
                let failBreakdown = Object.entries(data.metrics.failure_distribution || {})
                    .map(([k, v]) => `<li><strong>${k}</strong>: ${v} occurrence(s)</li>`)
                    .join("");

                telemetryDetail.innerHTML = `
                    <h4>Production Telemetry Overview</h4>
                    <p><strong>Total Queries Persisted:</strong> ${data.metrics.total_queries}</p>
                    <p><strong>Grounding Pass Rate:</strong> ${data.metrics.grounding_pass_rate}%</p>
                    <p><strong>Average Pipeline Latency:</strong> ${data.metrics.avg_latency_sec}s</p>
                    <p><strong>Average Grounding Confidence:</strong> ${Math.round(data.metrics.avg_confidence * 100)}%</p>
                    <p><strong>Self-Healing Recovery Rate:</strong> ${data.metrics.retry_success_rate}%</p>
                    <h5>Failure Diagnostics Distribution:</h5>
                    <ul>${failBreakdown || "<li>No failure records yet.</li>"}</ul>
                `;
            }
        } catch (e) {
            console.error("Failed to fetch telemetry", e);
        }
    }

    function renderResult(data) {
        // Update Telemetry Panel
        intentBadge.textContent = (data.intent || "RESEARCH_QUESTION").toUpperCase();
        intentBadge.className = "badge badge-neutral";

        const confPct = Math.round((data.grounding_confidence || 0) * 100);
        gaugeBar.style.width = `${confPct}%`;
        gaugeValue.textContent = `${confPct}%`;

        retryCount.textContent = data.retry_count || 0;
        if (data.retry_count > 0) {
            healingStatus.textContent = "Self-Healed";
            healingStatus.className = "badge badge-warning";
        } else {
            healingStatus.textContent = "Optimal";
            healingStatus.className = "badge badge-success";
        }

        failureType.textContent = (data.failure_type || "NONE").toUpperCase();
        latencyValue.textContent = `${data.latency_sec || 0}s`;

        // Render Grounded Badge & Answer
        if (data.is_grounded) {
            groundedBadge.textContent = "Fully Grounded";
            groundedBadge.className = "badge badge-success";
        } else {
            groundedBadge.textContent = "Partial Grounding";
            groundedBadge.className = "badge badge-warning";
        }

        // Format citations in answer text
        let formattedAnswer = (data.answer || "").replace(/\[([\w\.\-]+)\]/g, '<span class="citation-badge">arXiv:$1</span>');
        answerText.innerHTML = `<p>${formattedAnswer.replace(/\n/g, '<br>')}</p>`;

        // Render Evidence Chunks
        evidenceList.innerHTML = "";
        if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
            data.retrieved_chunks.forEach(chunk => {
                const card = document.createElement("div");
                card.className = "evidence-card";
                card.innerHTML = `
                    <div class="evidence-header">
                        <span class="paper-tag">arXiv:${chunk.paper_id} (Page ${chunk.page_number})</span>
                        <span class="score-tag">Score: ${chunk.reranker_score}</span>
                    </div>
                    <div class="evidence-text">${chunk.text}</div>
                `;
                evidenceList.appendChild(card);
            });
        } else {
            evidenceList.innerHTML = "<p class='text-secondary'>No retrieved chunks available.</p>";
        }

        // Render Self Healing Execution Log
        healingHistory.innerHTML = "";
        if (data.healing_history && data.healing_history.length > 0) {
            data.healing_history.forEach(plan => {
                const item = document.createElement("div");
                item.className = "history-item";
                item.innerHTML = `
                    <strong>Retry #${plan.retry_number}</strong>: Strategy <em>${plan.retrieval_strategy}</em> | Top-K=${plan.top_k} (Dense=${plan.dense_top_k}, BM25=${plan.bm25_top_k})<br>
                    <small style="color: var(--text-sub);">Reason: ${plan.reason}</small>
                `;
                healingHistory.appendChild(item);
            });
        } else {
            healingHistory.innerHTML = "<p class='text-secondary'>Direct answer executed without retries.</p>";
        }

        resultContainer.classList.remove("hidden");
    }
});
