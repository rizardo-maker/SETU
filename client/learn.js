/* ==========================================================================
   SETU LEARN — Client Controller & Accessible Audio Engine
   Features:
     - Document Ingestion & Section Navigator
     - Paragraph-by-Paragraph Accessible Speech Controller with Variable Speeds
     - Non-overlapping Audio Engine with `/api/tts`
     - Explain Simply, Multi-mode Summarize, Grounded RAG Ask, and Quiz Me
     - Full Screen Reader & Keyboard First Interaction
   ========================================================================== */

class SetuLearnApp {
  constructor() {
    this.doc = null;
    this.currentSectionIdx = 0;
    this.currentParaIdx = 0;
    this.readingSpeed = 1.0;
    this.isPlaying = false;
    this.currentAudio = null;
    this.speechAbortController = null;
    this._speechEpoch = 0;

    // Quiz State
    this.quizData = null;
    this.currentQuizIdx = 0;
    this.quizScore = 0;

    this._initDom();
    this._bindEvents();
    this._bindKeyboardShortcuts();
  }

  _initDom() {
    // Views
    this.viewUpload = document.getElementById("view-upload");
    this.viewStudy = document.getElementById("view-study");

    // Upload Elements
    this.dropzone = document.getElementById("upload-dropzone");
    this.fileInput = document.getElementById("file-input");
    this.btnBrowse = document.getElementById("btn-browse-file");
    this.uploadStatusCard = document.getElementById("upload-status-card");
    this.uploadStatusTitle = document.getElementById("upload-status-title");
    this.uploadStatusDetail = document.getElementById("upload-status-detail");

    // Study Elements
    this.docTitle = document.getElementById("doc-title");
    this.sectionSelect = document.getElementById("section-select");
    this.btnReadAloud = document.getElementById("btn-read-aloud");
    this.btnReadText = document.getElementById("btn-read-text");
    this.btnExplain = document.getElementById("btn-explain-simply");
    this.btnSummarize = document.getElementById("btn-summarize");
    this.btnAsk = document.getElementById("btn-ask-setu");
    this.btnQuiz = document.getElementById("btn-quiz-me");
    this.btnEndSession = document.getElementById("btn-end-session");

    // Reading & Stepper
    this.btnPrevSec = document.getElementById("btn-prev-section");
    this.btnNextSec = document.getElementById("btn-next-section");
    this.posIndicator = document.getElementById("reading-position-indicator");
    this.btnPrevPara = document.getElementById("btn-prev-para");
    this.btnRepeatPara = document.getElementById("btn-repeat-para");
    this.btnNextPara = document.getElementById("btn-next-para");
    this.currentParaText = document.getElementById("current-paragraph-text");
    this.paraCounter = document.getElementById("para-counter");
    this.activeParaCard = document.getElementById("active-paragraph-card");

    // Modals
    this.modalExplain = document.getElementById("modal-explain");
    this.explainContent = document.getElementById("explain-content");
    this.btnSpeakExplain = document.getElementById("btn-speak-explain");

    this.modalSummary = document.getElementById("modal-summarize");
    this.summaryContent = document.getElementById("summary-content");
    this.btnSpeakSummary = document.getElementById("btn-speak-summary");

    this.modalAsk = document.getElementById("modal-ask");
    this.askForm = document.getElementById("ask-form");
    this.askInput = document.getElementById("ask-input");
    this.askResultWrap = document.getElementById("ask-result-wrap");
    this.askAnswerText = document.getElementById("ask-answer-text");
    this.askSourceText = document.getElementById("ask-source-text");
    this.btnSpeakAnswer = document.getElementById("btn-speak-answer");

    this.modalQuiz = document.getElementById("modal-quiz");
    this.quizLoading = document.getElementById("quiz-loading");
    this.quizActive = document.getElementById("quiz-active");
    this.quizFinished = document.getElementById("quiz-finished");
    this.quizQIndicator = document.getElementById("quiz-q-indicator");
    this.quizScoreBadge = document.getElementById("quiz-score-badge");
    this.quizQuestionText = document.getElementById("quiz-question-text");
    this.quizOptionsContainer = document.getElementById("quiz-options-container");
    this.quizFeedbackBox = document.getElementById("quiz-feedback-box");
    this.quizFeedbackStatus = document.getElementById("quiz-feedback-status");
    this.quizFeedbackExplanation = document.getElementById("quiz-feedback-explanation");
    this.btnQuizNext = document.getElementById("btn-quiz-next");
    this.btnRestartQuiz = document.getElementById("btn-restart-quiz");

    this.modalShortcuts = document.getElementById("modal-shortcuts");
    this.srAnnouncer = document.getElementById("sr-announcer");
  }

  _announce(text) {
    if (!text) return;
    if (this.srAnnouncer) {
      this.srAnnouncer.textContent = text;
    }
  }

  _bindEvents() {
    // File Upload handling
    this.dropzone.addEventListener("click", () => this.fileInput.click());
    this.btnBrowse.addEventListener("click", (e) => {
      e.stopPropagation();
      this.fileInput.click();
    });
    this.dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        this.fileInput.click();
      }
    });

    this.fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        this._uploadFile(e.target.files[0]);
      }
    });

    // Drag & Drop
    this.dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      this.dropzone.classList.add("drag-over");
    });
    this.dropzone.addEventListener("dragleave", () => this.dropzone.classList.remove("drag-over"));
    this.dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      this.dropzone.classList.remove("drag-over");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        this._uploadFile(e.dataTransfer.files[0]);
      }
    });

    // Section Dropdown
    this.sectionSelect.addEventListener("change", (e) => {
      const idx = parseInt(e.target.value, 10);
      this._selectSection(idx);
    });

    // Primary Core Feature Triggers
    this.btnReadAloud.addEventListener("click", () => this._togglePlayPause());
    this.btnExplain.addEventListener("click", () => this._openExplainModal());
    this.btnSummarize.addEventListener("click", () => this._openSummarizeModal());
    this.btnAsk.addEventListener("click", () => this._openAskModal());
    this.btnQuiz.addEventListener("click", () => this._openQuizModal());

    // Navigation & Reading Steppers
    this.btnPrevSec.addEventListener("click", () => this._navigateSection(-1));
    this.btnNextSec.addEventListener("click", () => this._navigateSection(1));
    this.btnPrevPara.addEventListener("click", () => this._navigateParagraph(-1));
    this.btnNextPara.addEventListener("click", () => this._navigateParagraph(1));
    this.btnRepeatPara.addEventListener("click", () => this._readCurrentParagraph());

    // Speed Selector Chips
    document.querySelectorAll(".speed-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        document.querySelectorAll(".speed-chip").forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        this.readingSpeed = parseFloat(chip.getAttribute("data-speed") || "1.0");
        if (this.currentAudio) {
          this.currentAudio.playbackRate = this.readingSpeed;
        }
        this._announce(`Reading speed set to ${this.readingSpeed}x.`);
      });
    });

    // Modal Close Buttons
    document.getElementById("btn-close-explain").addEventListener("click", () => this._closeModal("explain"));
    document.getElementById("explain-backdrop").addEventListener("click", () => this._closeModal("explain"));

    document.getElementById("btn-close-summary").addEventListener("click", () => this._closeModal("summarize"));
    document.getElementById("summary-backdrop").addEventListener("click", () => this._closeModal("summarize"));

    document.getElementById("btn-close-ask").addEventListener("click", () => this._closeModal("ask"));
    document.getElementById("ask-backdrop").addEventListener("click", () => this._closeModal("ask"));

    document.getElementById("btn-close-quiz").addEventListener("click", () => this._closeModal("quiz"));
    document.getElementById("quiz-backdrop").addEventListener("click", () => this._closeModal("quiz"));

    document.getElementById("btn-shortcuts-modal").addEventListener("click", () => this._openModal("shortcuts"));
    document.getElementById("btn-close-shortcuts").addEventListener("click", () => this._closeModal("shortcuts"));
    document.getElementById("shortcuts-backdrop").addEventListener("click", () => this._closeModal("shortcuts"));

    // Modal Speech Buttons
    this.btnSpeakExplain.addEventListener("click", () => {
      const text = this.explainContent.textContent;
      this._speakDirect(text);
    });

    this.btnSpeakSummary.addEventListener("click", () => {
      const text = this.summaryContent.textContent;
      this._speakDirect(text);
    });

    this.btnSpeakAnswer.addEventListener("click", () => {
      const text = this.askAnswerText.textContent;
      this._speakDirect(text);
    });

    // Summary Tabs
    document.querySelectorAll(".summary-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".summary-tab").forEach((t) => {
          t.classList.remove("active");
          t.setAttribute("aria-selected", "false");
        });
        tab.classList.add("active");
        tab.setAttribute("aria-selected", "true");
        const mode = tab.getAttribute("data-mode") || "quick";
        this._fetchSummary(mode);
      });
    });

    // Ask Form
    this.askForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const q = this.askInput.value.trim();
      if (q) this._submitAskQuestion(q);
    });

    // Quiz Buttons
    this.btnQuizNext.addEventListener("click", () => this._nextQuizQuestion());
    this.btnRestartQuiz.addEventListener("click", () => this._generateQuizQuestions());

    // End Session Button
    this.btnEndSession.addEventListener("click", () => this._endSession());
  }

  _bindKeyboardShortcuts() {
    window.addEventListener("keydown", (e) => {
      // Ignore if user is currently typing in input / textarea
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") {
        if (e.key === "Escape") {
          e.target.blur();
        }
        return;
      }

      switch (e.key) {
        case " ":
          e.preventDefault();
          this._togglePlayPause();
          break;
        case "ArrowDown":
          e.preventDefault();
          this._navigateParagraph(1);
          break;
        case "ArrowUp":
          e.preventDefault();
          this._navigateParagraph(-1);
          break;
        case "r":
        case "R":
          e.preventDefault();
          this._readCurrentParagraph();
          break;
        case "e":
        case "E":
          e.preventDefault();
          this._openExplainModal();
          break;
        case "s":
        case "S":
          e.preventDefault();
          this._openSummarizeModal();
          break;
        case "q":
        case "Q":
          e.preventDefault();
          this._openAskModal();
          break;
        case "Escape":
          e.preventDefault();
          this._interruptSpeech();
          this._closeAllModals();
          break;
      }
    });
  }

  // ============================================================================
  // Document Upload & Ingestion
  // ============================================================================
  async _uploadFile(file) {
    if (!file) return;

    this.uploadStatusCard.classList.remove("hidden");
    this.uploadStatusTitle.textContent = "Uploading document…";
    this.uploadStatusDetail.textContent = `Reading ${file.name}`;
    this._announce("Uploading document.");

    const formData = new FormData();
    formData.append("file", file);

    try {
      this.uploadStatusTitle.textContent = "Reading document…";
      this.uploadStatusDetail.textContent = "Extracting headings and structure…";
      this._announce("Reading document.");

      const res = await fetch("/api/learn/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || "Document upload failed.");
      }

      this.uploadStatusTitle.textContent = "Document ready!";
      this.uploadStatusDetail.textContent = "Preparing accessible study console…";
      this._announce("Document ready.");

      setTimeout(() => {
        this._loadDocument(data.document);
      }, 500);
    } catch (err) {
      console.error("Upload error:", err);
      this.uploadStatusTitle.textContent = "Upload Failed";
      this.uploadStatusDetail.textContent = err.message || "Could not process document.";
      this._announce(`Error: ${err.message}`);
    }
  }

  _loadDocument(doc) {
    this.doc = doc;
    this.currentSectionIdx = 0;
    this.currentParaIdx = 0;

    this.docTitle.textContent = doc.title || "Study Material";
    this.btnEndSession.classList.remove("hidden");

    // Populate Sections Dropdown
    this.sectionSelect.innerHTML = "";
    doc.sections.forEach((sec, idx) => {
      const opt = document.createElement("option");
      opt.value = idx.toString();
      opt.textContent = `${sec.heading} (${sec.paragraphs.length} paragraphs)`;
      this.sectionSelect.appendChild(opt);
    });

    // Switch View to Study Console
    this.viewUpload.classList.add("hidden");
    this.viewStudy.classList.remove("hidden");

    this._selectSection(0);
    this._announce(`Document loaded: ${doc.title}. ${doc.sections.length} sections found.`);
  }

  _selectSection(secIdx) {
    if (!this.doc || !this.doc.sections[secIdx]) return;
    this._interruptSpeech();
    this.isPlaying = false;
    this._updatePlayButtonState();

    this.currentSectionIdx = secIdx;
    this.currentParaIdx = 0;
    this.sectionSelect.value = secIdx.toString();

    this._updateParagraphDisplay();
    const sec = this.doc.sections[secIdx];
    this._announce(`Section ${sec.heading} selected.`);
  }

  _navigateSection(delta) {
    if (!this.doc || !this.doc.sections.length) return;
    const nextIdx = this.currentSectionIdx + delta;
    if (nextIdx >= 0 && nextIdx < this.doc.sections.length) {
      this._selectSection(nextIdx);
    } else {
      this._announce(delta > 0 ? "You are on the last section." : "You are on the first section.");
    }
  }

  _navigateParagraph(delta) {
    if (!this.doc) return;
    const sec = this.doc.sections[this.currentSectionIdx];
    if (!sec || !sec.paragraphs.length) return;

    const nextParaIdx = this.currentParaIdx + delta;
    if (nextParaIdx >= 0 && nextParaIdx < sec.paragraphs.length) {
      this.currentParaIdx = nextParaIdx;
      this._updateParagraphDisplay();
      if (this.isPlaying) {
        this._readCurrentParagraph();
      }
    } else if (delta > 0 && this.currentSectionIdx + 1 < this.doc.sections.length) {
      // Automatically advance to next section if reading
      this._selectSection(this.currentSectionIdx + 1);
      if (this.isPlaying) {
        this._readCurrentParagraph();
      }
    } else {
      this._announce(delta > 0 ? "End of section reached." : "Beginning of section reached.");
      this.isPlaying = false;
      this._updatePlayButtonState();
    }
  }

  _updateParagraphDisplay() {
    if (!this.doc) return;
    const sec = this.doc.sections[this.currentSectionIdx];
    if (!sec || !sec.paragraphs.length) {
      this.currentParaText.textContent = "No text in this section.";
      this.posIndicator.textContent = "Section Empty";
      return;
    }

    const para = sec.paragraphs[this.currentParaIdx] || "";
    this.currentParaText.textContent = para;
    this.paraCounter.textContent = `Paragraph ${this.currentParaIdx + 1} of ${sec.paragraphs.length}`;
    this.posIndicator.textContent = `Section ${this.currentSectionIdx + 1}/${this.doc.sections.length} • Para ${this.currentParaIdx + 1}/${sec.paragraphs.length}`;
  }

  // ============================================================================
  // Accessible Non-Overlapping Audio Engine with `/api/tts`
  // ============================================================================
  _interruptSpeech() {
    this._speechEpoch++;
    if (this.currentAudio) {
      try {
        this.currentAudio.pause();
        this.currentAudio.src = "";
      } catch (_) {}
      this.currentAudio = null;
    }
  }

  _updatePlayButtonState() {
    if (this.isPlaying) {
      this.btnReadAloud.classList.add("btn-playing");
      this.btnReadText.textContent = "Pause Reading";
      this.btnReadAloud.setAttribute("aria-label", "Pause audio reading (Space)");
    } else {
      this.btnReadAloud.classList.remove("btn-playing");
      this.btnReadText.textContent = "Read Aloud";
      this.btnReadAloud.setAttribute("aria-label", "Start audio reading (Space)");
    }
  }

  _togglePlayPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this._interruptSpeech();
      this._updatePlayButtonState();
      this._announce("Reading paused.");
    } else {
      this.isPlaying = true;
      this._updatePlayButtonState();
      this._announce("Reading resumed.");
      this._readCurrentParagraph();
    }
  }

  async _readCurrentParagraph() {
    if (!this.doc) return;
    const sec = this.doc.sections[this.currentSectionIdx];
    if (!sec || !sec.paragraphs[this.currentParaIdx]) return;

    const textToRead = sec.paragraphs[this.currentParaIdx].trim();
    if (!textToRead) {
      this._navigateParagraph(1);
      return;
    }

    const epoch = ++this._speechEpoch;
    this._interruptSpeech();

    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToRead }),
      });

      if (epoch !== this._speechEpoch) return;
      if (!res.ok) throw new Error("TTS failed");

      const blob = await res.blob();
      if (epoch !== this._speechEpoch) return;

      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);
      this.currentAudio = audio;
      audio.playbackRate = this.readingSpeed;

      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        if (epoch === this._speechEpoch && this.isPlaying) {
          // Advance to next paragraph sequentially
          this._navigateParagraph(1);
        }
      };

      audio.onerror = () => {
        URL.revokeObjectURL(audioUrl);
      };

      await audio.play();
    } catch (err) {
      console.warn("Audio speech error:", err);
    }
  }

  async _speakDirect(text) {
    if (!text || !text.trim()) return;
    this._interruptSpeech();
    const epoch = ++this._speechEpoch;

    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim() }),
      });

      if (epoch !== this._speechEpoch) return;
      if (!res.ok) return;

      const blob = await res.blob();
      if (epoch !== this._speechEpoch) return;

      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);
      this.currentAudio = audio;
      audio.playbackRate = this.readingSpeed;
      audio.onended = () => URL.revokeObjectURL(audioUrl);
      await audio.play();
    } catch (err) {
      console.warn("Direct speak error:", err);
    }
  }

  // ============================================================================
  // Feature 1: Explain Simply
  // ============================================================================
  async _openExplainModal() {
    if (!this.doc) return;
    this._interruptSpeech();
    this.isPlaying = false;
    this._updatePlayButtonState();

    this._openModal("explain");
    this.explainContent.textContent = "Generating simple explanation with local AI…";
    this._announce("Generating plain language explanation.");

    const sec = this.doc.sections[this.currentSectionIdx];
    try {
      const res = await fetch("/api/learn/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: this.doc.document_id,
          section_id: sec ? sec.section_id : null,
        }),
      });

      const data = await res.json();
      const text = data.explanation || "Could not explain this section.";
      this.explainContent.textContent = text;
      this._announce("Explanation ready. Press space or click read explanation.");
    } catch (err) {
      this.explainContent.textContent = "The learning assistant is temporarily unavailable.";
      this._announce("Explanation unavailable.");
    }
  }

  // ============================================================================
  // Feature 2: Summarize
  // ============================================================================
  async _openSummarizeModal() {
    if (!this.doc) return;
    this._interruptSpeech();
    this.isPlaying = false;
    this._updatePlayButtonState();

    this._openModal("summarize");
    this._fetchSummary("quick");
  }

  async _fetchSummary(mode) {
    if (!this.doc) return;
    this.summaryContent.textContent = `Generating ${mode.replace('_', ' ')} summary…`;
    this._announce(`Generating ${mode.replace('_', ' ')} summary.`);

    const sec = this.doc.sections[this.currentSectionIdx];
    try {
      const res = await fetch("/api/learn/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: this.doc.document_id,
          section_id: sec ? sec.section_id : null,
          mode: mode,
        }),
      });

      const data = await res.json();
      const text = data.summary || "Could not generate summary.";
      this.summaryContent.textContent = text;
      this._announce("Summary ready.");
    } catch (err) {
      this.summaryContent.textContent = "The learning assistant is temporarily unavailable.";
      this._announce("Summary unavailable.");
    }
  }

  // ============================================================================
  // Feature 3: Ask SETU (Grounded RAG)
  // ============================================================================
  _openAskModal() {
    if (!this.doc) return;
    this._interruptSpeech();
    this.isPlaying = false;
    this._updatePlayButtonState();

    this._openModal("ask");
    this.askInput.value = "";
    this.askResultWrap.classList.add("hidden");
    this.btnSpeakAnswer.classList.add("hidden");
    setTimeout(() => this.askInput.focus(), 150);
  }

  async _submitAskQuestion(question) {
    if (!this.doc || !question) return;

    this.askResultWrap.classList.remove("hidden");
    this.askAnswerText.textContent = "Searching document and generating answer…";
    this.askSourceText.textContent = "Searching…";
    this.btnSpeakAnswer.classList.add("hidden");
    this._announce("Searching uploaded material.");

    try {
      const res = await fetch("/api/learn/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: this.doc.document_id,
          question: question,
        }),
      });

      const data = await res.json();
      this.askAnswerText.textContent = data.answer || "This information was not found in the uploaded material.";
      
      if (data.source) {
        this.askSourceText.textContent = data.source;
        document.getElementById("ask-source-badge").classList.remove("hidden");
      } else {
        document.getElementById("ask-source-badge").classList.add("hidden");
      }

      this.btnSpeakAnswer.classList.remove("hidden");
      this._announce(`Answer: ${data.answer}`);
      this._speakDirect(data.answer);
    } catch (err) {
      this.askAnswerText.textContent = "The learning assistant is temporarily unavailable.";
      this._announce("Answer unavailable.");
    }
  }

  // ============================================================================
  // Feature 4: Quiz Me (Interactive MCQs)
  // ============================================================================
  async _openQuizModal() {
    if (!this.doc) return;
    this._interruptSpeech();
    this.isPlaying = false;
    this._updatePlayButtonState();

    this._openModal("quiz");
    this._generateQuizQuestions();
  }

  async _generateQuizQuestions() {
    this.quizLoading.classList.remove("hidden");
    this.quizActive.classList.add("hidden");
    this.quizFinished.classList.add("hidden");
    this._announce("Generating quiz questions from notes.");

    try {
      const res = await fetch("/api/learn/quiz", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: this.doc.document_id,
          num_questions: 5,
        }),
      });

      const data = await res.json();
      if (!data.questions || !data.questions.length) {
        throw new Error("Could not generate quiz.");
      }

      this.quizData = data.questions;
      this.currentQuizIdx = 0;
      this.quizScore = 0;

      this.quizLoading.classList.add("hidden");
      this.quizActive.classList.remove("hidden");
      this._renderQuizQuestion();
    } catch (err) {
      this.quizLoading.innerHTML = `<p class="error-msg">I couldn't create the quiz. Please try again.</p>`;
      this._announce("I couldn't create the quiz. Please try again.");
    }
  }

  _renderQuizQuestion() {
    if (!this.quizData || !this.quizData[this.currentQuizIdx]) return;
    const q = this.quizData[this.currentQuizIdx];

    this.quizQIndicator.textContent = `Question ${this.currentQuizIdx + 1} of ${this.quizData.length}`;
    this.quizScoreBadge.textContent = `Score: ${this.quizScore}`;
    this.quizQuestionText.textContent = q.question;
    this.quizFeedbackBox.classList.add("hidden");

    this.quizOptionsContainer.innerHTML = "";
    const letters = ["A", "B", "C", "D"];

    q.options.forEach((opt, idx) => {
      const btn = document.createElement("button");
      btn.className = "quiz-option-btn";
      btn.type = "button";
      btn.setAttribute("role", "radio");
      btn.setAttribute("aria-checked", "false");
      btn.innerHTML = `<span class="opt-letter">${letters[idx]}</span> <span class="opt-text">${opt}</span>`;
      btn.addEventListener("click", () => this._selectQuizAnswer(idx, q.correct_index, q.explanation));
      this.quizOptionsContainer.appendChild(btn);
    });

    this._announce(`Question ${this.currentQuizIdx + 1}: ${q.question}`);
  }

  _selectQuizAnswer(selectedIdx, correctIdx, explanation) {
    const btns = this.quizOptionsContainer.querySelectorAll(".quiz-option-btn");
    btns.forEach((btn, idx) => {
      btn.disabled = true;
      if (idx === correctIdx) {
        btn.classList.add("opt-correct");
      }
      if (idx === selectedIdx && idx !== correctIdx) {
        btn.classList.add("opt-incorrect");
      }
    });

    const isCorrect = selectedIdx === correctIdx;
    if (isCorrect) {
      this.quizScore++;
      this.quizScoreBadge.textContent = `Score: ${this.quizScore}`;
      this.quizFeedbackStatus.textContent = "✅ Correct!";
      this.quizFeedbackStatus.className = "feedback-status status-correct";
      this._announce("Correct answer.");
      this._speakDirect(`Correct. ${explanation}`);
    } else {
      this.quizFeedbackStatus.textContent = "❌ Incorrect";
      this.quizFeedbackStatus.className = "feedback-status status-incorrect";
      this._announce("Incorrect answer.");
      this._speakDirect(`Incorrect. ${explanation}`);
    }

    this.quizFeedbackExplanation.textContent = explanation;
    this.quizFeedbackBox.classList.remove("hidden");
    this.btnQuizNext.focus();
  }

  _nextQuizQuestion() {
    this.currentQuizIdx++;
    if (this.currentQuizIdx < this.quizData.length) {
      this._renderQuizQuestion();
    } else {
      this._finishQuiz();
    }
  }

  _finishQuiz() {
    this.quizActive.classList.add("hidden");
    this.quizFinished.classList.remove("hidden");
    const total = this.quizData.length;
    const finalMsg = `Quiz complete! You scored ${this.quizScore} out of ${total}.`;
    document.getElementById("quiz-final-score").textContent = finalMsg;
    this._announce(finalMsg);
    this._speakDirect(finalMsg);
  }

  // ============================================================================
  // Modal Utilities & Session End
  // ============================================================================
  _openModal(name) {
    this._closeAllModals();
    const map = {
      explain: this.modalExplain,
      summarize: this.modalSummary,
      ask: this.modalAsk,
      quiz: this.modalQuiz,
      shortcuts: this.modalShortcuts,
    };
    const modal = map[name];
    if (modal) {
      modal.classList.remove("hidden");
    }
  }

  _closeModal(name) {
    const map = {
      explain: this.modalExplain,
      summarize: this.modalSummary,
      ask: this.modalAsk,
      quiz: this.modalQuiz,
      shortcuts: this.modalShortcuts,
    };
    const modal = map[name];
    if (modal) modal.classList.add("hidden");
  }

  _closeAllModals() {
    this.modalExplain.classList.add("hidden");
    this.modalSummary.classList.add("hidden");
    this.modalAsk.classList.add("hidden");
    this.modalQuiz.classList.add("hidden");
    this.modalShortcuts.classList.add("hidden");
  }

  async _endSession() {
    this._interruptSpeech();
    if (this.doc) {
      try {
        await fetch(`/api/learn/document/${this.doc.document_id}`, { method: "DELETE" });
      } catch (_) {}
    }
    this.doc = null;
    this.viewStudy.classList.add("hidden");
    this.viewUpload.classList.remove("hidden");
    this.btnEndSession.classList.add("hidden");
    this.uploadStatusCard.classList.add("hidden");
    this.fileInput.value = "";
    this._announce("Session ended. Upload a new study document.");
  }
}

// Instantiate on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  window.setuLearn = new SetuLearnApp();
});
