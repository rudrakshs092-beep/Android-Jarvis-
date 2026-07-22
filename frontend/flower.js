function Draw() {
  frames--;
  ctx.clearRect(0, 0, 2 * Cw, 2 * Ch);
  t = frames * rad;
  rx = R x * Math.abs(Math.cos(t)) + 50;
  ry = R y * Math.abs(Math.sin(t)) + 50;

  x = cx + rx * Math.sin(kx * t + Math.PI / 2);
  y = cy + ry * Math.sin(ky * t + Math.PI / 2);

  x1 = cx + rx * Math.sin(kx * t + Math.PI);
  y1 = cy + ry * Math.sin(ky * t + Math.PI);

  x2 = cx + rx * Math.sin(kx * t);
  y2 = cy + ry * Math.sin(ky * t);

  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.quadraticCurveTo(x1, y1, x2, y2);
  ctx.stroke();
  ctx.globalCompositeOperation = "lighter";
}
