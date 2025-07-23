// Copyright (c) 2025 Salvador E. Tropea
// Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
// License: GPLv3
// Project: ComfyUI-AudioBatch

// This script adds an event named "set-audiobatch-toast"
// Used to notify the user in the GUI using the Toast API

import { app } from "/scripts/app.js";

try {
    // Register a new extension
    app.registerExtension({
        name: "SET.SeCoNoHe.ToastHandler",  // Unique name

        // The setup function is executed when the extension is loaded
        setup() {
            // Add a listener for our custom event
            app.api.addEventListener("seconohe-toast", (event) => {
                // The data from Python is in event.detail
                const { message, summary, severity } = event.detail;

                // Use the ComfyUI toast API to show the message
                // app.ui.toast.addMessage is the modern way to do this
                app.extensionManager.toast.add({
                    severity: severity,
                    summary: summary,
                    detail: message,
                    life: 6000
                });
            });
        },
    });
} catch (e) {
    // If an error is caught, check if it's the "already registered" error
    if (e.message && e.message.includes("already registered")) {
        // This is expected if the script is loaded more than once.
        // We can safely ignore it and log a message for debugging.
        console.log("SET.SeCoNoHe.ToastHandler was already registered. Ignoring duplicate registration.");
    } else {
        // If it's a different error, re-throw it so it appears in the console
        // and we know something else went wrong.
        console.error("An unexpected error occurred during ToastHandler registration:", e);
        throw e;
    }
}
