// Color Theme: Neon Cyan / Arc Reactor Blue (#00f3ff)
const ThemeColor = "#00f3ff";

function drawIronManFace(ctx, width, height, pulseFactor) {
  const cx = width / 2;
  const cy = height / 2;

  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = ThemeColor;
  ctx.fillStyle = ThemeColor;
  ctx.shadowColor = ThemeColor;
  ctx.shadowBlur = 15 * pulseFactor; // Pulse effect jab JARVIS baat kare
  ctx.lineWidth = 2;

  // Outer Mask Contour (Iron Man Helmet Shape)
  ctx.beginPath();
  ctx.moveTo(cx - 60, cy - 80);
  ctx.lineTo(cx + 60, cy - 80);
  ctx.lineTo(cx + 80, cy - 20);
  ctx.lineTo(cx + 50, cy + 80);
  ctx.lineTo(cx + 30, cy + 100);
  ctx.lineTo(cx - 30, cy + 100);
  ctx.lineTo(cx - 50, cy + 80);
  ctx.lineTo(cx - 80, cy - 20);
  ctx.closePath();
  ctx.stroke();

  // Glowing Eyes (Cyan/Light Blue)
  ctx.beginPath();
  // Left Eye
  ctx.moveTo(cx - 45, cy - 15);
  ctx.lineTo(cx - 15, cy - 15);
  ctx.lineTo(cx - 25, cy - 5);
  ctx.closePath();
  // Right Eye
  ctx.moveTo(cx + 45, cy - 15);
  ctx.lineTo(cx + 15, cy - 15);
  ctx.lineTo(cx + 25, cy - 5);
  ctx.closePath();
  ctx.fill();

  // Jaw Line Detailing
  ctx.beginPath();
  ctx.moveTo(cx - 40, cy + 20);
  ctx.lineTo(cx - 20, cy + 60);
  ctx.lineTo(cx + 20, cy + 60);
  ctx.lineTo(cx + 40, cy + 20);
  ctx.stroke();
}
