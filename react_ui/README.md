## Running this project (quick start)

These commands assume you're on Windows using PowerShell. Open a PowerShell terminal in the `react_ui` folder and run the commands below.

1. Install dependencies (only once):

```powershell
npm install
```

2. Start the dev server with HMR:

```powershell
npm run dev
```
python src/ocr_service.py

3. Build for production:

```powershell
npm run build
```

4. Preview the production build locally:

```powershell
npm run preview
```

Notes and troubleshooting
- If `npm run dev` fails, ensure you have Node.js (v16 or later recommended) and npm installed. Check with `node -v` and `npm -v`.
- If ports are in use, Vite will prompt to use another port. You can also set the port in `vite.config.js`.
- PowerShell users: if scripts fail due to execution policy, run PowerShell as Administrator and set an appropriate policy or run commands individually.
- If you want to serve the `dist` folder with a simple static server, you can install `serve` globally: `npm i -g serve` and then `serve -s dist`.

If you'd like, I can add a short section describing how to integrate the background image (from the example screenshot) and fine-tune the glass effect.
