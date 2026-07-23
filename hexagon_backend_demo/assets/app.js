(() => {
  const slides = Array.from(document.querySelectorAll(".slide"));
  const nav = document.getElementById("slideNav");
  const progress = document.getElementById("progress");
  const title = document.getElementById("toolbarTitle");
  const notes = document.getElementById("notesDrawer");
  const notesText = document.getElementById("notesText");
  let current = 0;

  const clamp = (value) => Math.max(0, Math.min(slides.length - 1, value));

  function hashIndex() {
    const raw = window.location.hash.replace(/^#/, "");
    const parsed = Number.parseInt(raw, 10);
    return Number.isFinite(parsed) ? clamp(parsed - 1) : 0;
  }

  function buildNav() {
    slides.forEach((slide, index) => {
      slide.dataset.page = `${String(index + 1).padStart(2, "0")} / ${String(slides.length).padStart(2, "0")}`;
      const button = document.createElement("button");
      button.type = "button";
      button.innerHTML = `<span class="num">${String(index + 1).padStart(2, "0")}</span><span class="label">${slide.dataset.title}</span>`;
      button.addEventListener("click", () => show(index));
      nav.appendChild(button);
    });
  }

  function show(index, updateHash = true) {
    current = clamp(index);
    slides.forEach((slide, i) => slide.classList.toggle("active", i === current));
    Array.from(nav.children).forEach((button, i) => button.classList.toggle("active", i === current));
    title.innerHTML = `<strong>${String(current + 1).padStart(2, "0")}</strong> / ${String(slides.length).padStart(2, "0")} &nbsp; ${slides[current].dataset.title}`;
    progress.style.width = `${((current + 1) / slides.length) * 100}%`;
    notesText.textContent = slides[current].dataset.notes || "本页无补充讲稿。";
    nav.children[current]?.scrollIntoView({ block: "nearest" });
    document.body.classList.remove("nav-open");
    if (updateHash) {
      history.replaceState(null, "", `#${current + 1}`);
    }
  }

  function toggleNotes() {
    notes.classList.toggle("open");
  }

  function toggleFullscreen() {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      document.documentElement.requestFullscreen();
    }
  }

  document.getElementById("prevButton").addEventListener("click", () => show(current - 1));
  document.getElementById("nextButton").addEventListener("click", () => show(current + 1));
  document.getElementById("notesButton").addEventListener("click", toggleNotes);
  document.getElementById("fullscreenButton").addEventListener("click", toggleFullscreen);
  document.getElementById("printButton").addEventListener("click", () => window.print());
  document.getElementById("menuButton").addEventListener("click", () => document.body.classList.toggle("nav-open"));

  document.addEventListener("keydown", (event) => {
    if (["ArrowRight", "ArrowDown", "PageDown", " "].includes(event.key)) {
      event.preventDefault();
      show(current + 1);
    } else if (["ArrowLeft", "ArrowUp", "PageUp"].includes(event.key)) {
      event.preventDefault();
      show(current - 1);
    } else if (event.key === "Home") {
      show(0);
    } else if (event.key === "End") {
      show(slides.length - 1);
    } else if (event.key.toLowerCase() === "f") {
      toggleFullscreen();
    } else if (event.key.toLowerCase() === "n") {
      toggleNotes();
    } else if (event.key === "Escape") {
      notes.classList.remove("open");
      document.body.classList.remove("nav-open");
    }
  });

  document.querySelectorAll(".copy-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const target = document.getElementById(button.dataset.copyTarget);
      if (!target) return;
      const clone = target.cloneNode(true);
      clone.querySelectorAll(".copy-button").forEach((item) => item.remove());
      const value = clone.textContent.trim();
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(value);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = value;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        textarea.remove();
      }
      const old = button.textContent;
      button.textContent = "已复制";
      window.setTimeout(() => {
        button.textContent = old;
      }, 1200);
    });
  });

  window.addEventListener("hashchange", () => show(hashIndex(), false));
  buildNav();
  show(hashIndex(), false);
})();
