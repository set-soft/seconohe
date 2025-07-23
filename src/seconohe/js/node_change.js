// Copyright (c) 2025 Salvador E. Tropea
// Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
// License: GPLv3
// Project: SeCoNoHe

// This script adds an event named "set-audioseparation-node"
// It can currently just modify a widget value for the current node

import { app } from "/scripts/app.js";

try {
    // Register a new extension
    app.registerExtension({
        name: "SET.SeCoNoHe.NodeAdjust",  // Unique name

        // The setup function is executed when the extension is loaded
        setup() {
            // Add a listener for our custom event
            app.api.addEventListener("seconohe-node", (event) => {
                // The data from Python is in event.detail
                const { action, arg1, arg2 } = event.detail;

                // Find the node that is currently being executed
                const node = app.graph.getNodeById(app.runningNodeId);
                if (!node) {
                    console.warn(`[SET.SeCoNoHe] Could not find running node with ID: ${app.runningNodeId}`);
                    return;
                }

                // --- ACTION EXECUTED HERE ---
                switch (action) {
                    case 'change_widget':
                        // arg1 = widget name (e.g., "model")
                        // arg2 = new value (e.g., "&#x0001F4BE; My Awesome Model")

                        const widget = node.widgets.find(w => w.name === arg1);
                        if (widget) {
                            // This is the key part for combo boxes (dropdowns)
                            // If the new value isn't in the list of options, add it first.
                            if (!widget.options.values.includes(arg2)) {
                                widget.options.values.push(arg2);
                            }

                            // Set the widget value
                            widget.setValue(arg2, node, app.canvas);
                        } else {
                            console.error(`[SET.SeCoNoHe] Widget '${arg1}' not found on node ${node.id}`);
                        }
                        break;

                    // Other actions here in the future
                    // case 'disable_widget':
                    //     ...
                    //     break;
                }
            });
        },
    });
} catch (e) {
    // If an error is caught, check if it's the "already registered" error
    if (e.message && e.message.includes("already registered")) {
        // This is expected if the script is loaded more than once.
        // We can safely ignore it and log a message for debugging.
        console.log("SET.SeCoNoHe.NodeAdjust was already registered. Ignoring duplicate registration.");
    } else {
        // If it's a different error, re-throw it so it appears in the console
        // and we know something else went wrong.
        console.error("An unexpected error occurred during NodeAdjust registration:", e);
        throw e;
    }
}
