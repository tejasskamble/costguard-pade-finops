(function initCinematicVfx() {
  let rootDoc = document;
  try {
    if (window.parent && window.parent.document) {
      rootDoc = window.parent.document;
    }
  } catch (err) {
    rootDoc = document;
  }

  if (rootDoc.getElementById("cg-vfx-canvas")) {
    return;
  }

  const canvas = rootDoc.createElement("canvas");
  canvas.id = "cg-vfx-canvas";
  canvas.setAttribute("aria-hidden", "true");
  Object.assign(canvas.style, {
    position: "fixed",
    inset: "0",
    zIndex: "0",
    pointerEvents: "none",
    opacity: "0.25",
  });
  rootDoc.body.prepend(canvas);

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  const particles = [];
  const particleCount = 56;

  function resize() {
    const width = rootDoc.defaultView ? rootDoc.defaultView.innerWidth : window.innerWidth;
    const height = rootDoc.defaultView ? rootDoc.defaultView.innerHeight : window.innerHeight;
    canvas.width = width;
    canvas.height = height;
  }

  function spawn() {
    particles.length = 0;
    for (let i = 0; i < particleCount; i += 1) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.22,
        vy: (Math.random() - 0.5) * 0.22,
        r: Math.random() * 1.7 + 0.3,
        hue: Math.random() > 0.5 ? 188 : 260,
        a: Math.random() * 0.5 + 0.1,
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

      ctx.beginPath();
      ctx.fillStyle = `hsla(${p.hue}, 100%, 70%, ${p.a})`;
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }

    for (let i = 0; i < particles.length; i += 1) {
      const a = particles[i];
      for (let j = i + 1; j < particles.length; j += 1) {
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < 115) {
          const alpha = (1 - d / 115) * 0.16;
          ctx.strokeStyle = `rgba(0, 229, 255, ${alpha})`;
          ctx.lineWidth = 0.6;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    window.requestAnimationFrame(draw);
  }

  resize();
  spawn();
  draw();
  window.addEventListener("resize", () => {
    resize();
    spawn();
  });
})();
