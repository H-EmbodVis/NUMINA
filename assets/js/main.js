(() => {
  const highlightPromptNumerals = () => {
    const numeralRegex =
      /\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b/gi;
    const prompts = document.querySelectorAll(".pair-prompt");

    prompts.forEach((prompt) => {
      const walker = document.createTreeWalker(prompt, NodeFilter.SHOW_TEXT);
      const textNodes = [];

      while (walker.nextNode()) {
        const node = walker.currentNode;
        if (!node.nodeValue.trim()) continue;
        if (node.parentElement && node.parentElement.closest(".num-highlight")) {
          continue;
        }
        textNodes.push(node);
      }

      textNodes.forEach((node) => {
        const text = node.nodeValue;
        numeralRegex.lastIndex = 0;
        if (!numeralRegex.test(text)) return;

        numeralRegex.lastIndex = 0;
        let lastIndex = 0;
        const frag = document.createDocumentFragment();
        let match;

        while ((match = numeralRegex.exec(text)) !== null) {
          const start = match.index;
          const end = start + match[0].length;

          if (start > lastIndex) {
            frag.appendChild(document.createTextNode(text.slice(lastIndex, start)));
          }

          const mark = document.createElement("span");
          mark.className = "num-highlight";
          mark.textContent = match[0];
          frag.appendChild(mark);

          lastIndex = end;
        }

        if (lastIndex < text.length) {
          frag.appendChild(document.createTextNode(text.slice(lastIndex)));
        }

        node.parentNode.replaceChild(frag, node);
      });
    });
  };

  highlightPromptNumerals();

  const ensureAutoplay = () => {
    const videos = document.querySelectorAll(".demo-video");
    videos.forEach((video) => {
      video.muted = true;
      video.autoplay = true;
      video.loop = true;
      video.playsInline = true;

      const tryPlay = () => {
        const playPromise = video.play();
        if (playPromise && typeof playPromise.catch === "function") {
          playPromise.catch(() => {});
        }
      };

      video.addEventListener("canplay", tryPlay, { once: true });
      tryPlay();
    });
  };

  ensureAutoplay();

  const revealTargets = document.querySelectorAll(
    ".card, h2, .hero .subtitle, .hero .button-row"
  );
  revealTargets.forEach((el) => el.setAttribute("data-reveal", ""));

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );

  revealTargets.forEach((el) => observer.observe(el));
})();
