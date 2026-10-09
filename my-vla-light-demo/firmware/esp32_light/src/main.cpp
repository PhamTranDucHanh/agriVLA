#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>

// Copy config.example.h to config.h (ignored by Git) before upload.
#if __has_include("config.h")
#include "config.h"
#else
#include "config.example.h"
#endif

WebServer server(80);
bool lightOn = false;

void handleGetState() {
    server.send(200, "application/json",
                lightOn ? "{\"light_on\":true}" : "{\"light_on\":false}");
}

void handleSetLight() {
    String command = server.arg("plain");
    command.trim();
    command.toUpperCase();
    if (command != "ON" && command != "OFF") {
        server.send(400, "application/json", "{\"error\":\"Expected ON or OFF\"}");
        return;
    }
    bool requestedState = command == "ON";
    bool changed = requestedState != lightOn;
    if (changed) {
        digitalWrite(LED_PIN, requestedState ? HIGH : LOW);
        lightOn = requestedState;
    }
    Serial.printf("Command: %s | Light: %s | Changed: %s\n",
                  command.c_str(), lightOn ? "ON" : "OFF", changed ? "YES" : "NO");
    String response = "{\"light_on\":";
    response += lightOn ? "true" : "false";
    response += ",\"changed\":";
    response += changed ? "true" : "false";
    response += "}";
    server.send(200, "application/json", response);
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.println("Connecting to Wi-Fi; verify local config.h if connection fails.");
    server.on("/state", HTTP_GET, handleGetState);
    server.on("/light", HTTP_POST, handleSetLight);
    server.onNotFound([]() {
        server.send(404, "application/json", "{\"error\":\"Not found\"}");
    });
    server.begin();
}

void loop() {
    static bool wasConnected = false;
    bool connected = WiFi.status() == WL_CONNECTED;
    if (connected != wasConnected) {
        if (connected) {
            Serial.print("Wi-Fi connected! ESP32 IP: ");
            Serial.println(WiFi.localIP());
            Serial.println("HTTP server started");
        } else {
            Serial.println("Wi-Fi disconnected; reconnecting. GPIO state retained.");
        }
        wasConnected = connected;
    }
    server.handleClient();
    delay(2);
}
