using System;
using System.Text;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using Valve.VR;
using TMPro;
using UnityEngine.UI;
using System.Collections.Generic;
using System.IO;

public class TextOverlay : MonoBehaviour
{
    // Message class to track timing and fade
    private class TimedMessage
    {
        public string Text { get; set; }
        public float TimeReceived { get; set; }
        public float Alpha { get; set; }

        public TimedMessage(string text)
        {
            Text = text;
            TimeReceived = Time.time;
            Alpha = 1.0f;
        }
    }

    [Header("Socket Settings")]
    public string host = "127.0.0.1";
    public int port = 65432;

    [Header("Overlay Settings")]
    public float width = 1.0f;
    public float height = 0.5f;
    public Vector3 position = new Vector3(0, 1.6f, -1f);
    public Vector3 rotation = new Vector3(0, 0, 0);

    [Header("Message Settings")]
    public float messageFadeStartTime = 7.0f;
    public float messageFadeDuration = 3.0f;
    public float fadeUpdateInterval = 0.05f; // Reduced interval for smoother fading

    [Header("TextMeshPro Settings")]
    [Tooltip("Assign the TMP Font Asset for the TextMeshProUGUI component.")]
    public TMP_FontAsset fontAsset;

    private ulong overlayHandle = OpenVR.k_ulOverlayHandleInvalid;
    private TcpClient client;
    private NetworkStream stream;
    private Thread receiveThread;
    private volatile bool keepReading = false;

    private StringBuilder messageBuilder = new StringBuilder();
    private object lockObject = new object();

    // For rendering text to texture
    private RenderTexture renderTexture;
    private Camera textCamera;
    private TextMeshProUGUI textMesh;
    private Texture_t overlayTexture;

    private Queue<TimedMessage> lastSentences = new Queue<TimedMessage>();
    private const int MAX_SENTENCES = 10;

    // Thread-safe queue for incoming messages
    private Queue<string> pendingMessages = new Queue<string>();

    private StreamReader reader;

    // Flag to indicate if a render is needed
    private bool needsRender = false;

    private Coroutine fadeCoroutine; // Reference to the fade coroutine

    void Start()
    {
        InitializeOverlay();
        ConnectToPythonServer();
        fadeCoroutine = StartCoroutine(FadeMessagesRoutine()); // Start the fade coroutine
        Application.targetFrameRate = 30; // Optional: Cap frame rate to reduce GPU load
    }

    private void InitializeOverlay()
    {
        // Initialize OpenVR if needed
        if (OpenVR.System == null)
        {
            var error = EVRInitError.None;
            OpenVR.Init(ref error, EVRApplicationType.VRApplication_Overlay);
            if (error != EVRInitError.None)
            {
                Debug.LogError($"Failed to initialize OpenVR: {error}");
                return;
            }
        }

        // Create overlay
        var error2 = OpenVR.Overlay.CreateOverlay("TextOverlayKey", "Text Overlay", ref overlayHandle);
        if (error2 != EVROverlayError.None)
        {
            Debug.LogError($"Failed to create overlay: {error2}");
            return;
        }

        // Adjust render texture with more vertical space
        renderTexture = new RenderTexture(2048, 1536, 24);
        renderTexture.antiAliasing = 1;
        renderTexture.Create();

        // Create camera for rendering text
        var cameraObj = new GameObject("TextCamera");
        textCamera = cameraObj.AddComponent<Camera>();
        textCamera.clearFlags = CameraClearFlags.SolidColor;
        textCamera.backgroundColor = new Color(0, 0, 0, 0);
        textCamera.orthographic = true;
        textCamera.orthographicSize = 7;
        textCamera.targetTexture = renderTexture;
        textCamera.enabled = true;

        // Create Canvas with proper dimensions
        var canvasObj = new GameObject("OverlayCanvas");
        var canvas = canvasObj.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        canvas.worldCamera = textCamera;
        canvasObj.transform.SetParent(cameraObj.transform, false);
        
        var canvasRect = canvasObj.GetComponent<RectTransform>();
        canvasRect.sizeDelta = new Vector2(2200, 3000); // Adjusted width to be 30% narrower
        
        var scaler = canvasObj.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ConstantPixelSize;
        scaler.scaleFactor = 1;

        // Create text object as UI element
        var textObj = new GameObject("OverlayText");
        textObj.transform.SetParent(canvasObj.transform, false);
        textMesh = textObj.AddComponent<TextMeshProUGUI>();
        
        textMesh.enableWordWrapping = true;
        textMesh.horizontalAlignment = HorizontalAlignmentOptions.Center;
        textMesh.verticalAlignment = VerticalAlignmentOptions.Bottom;
        textMesh.fontSize = 48;
        textMesh.color = Color.white;
        textMesh.margin = new Vector4(20, 10, 20, 10);
        textMesh.lineSpacing = 4;

        // Assign the selected font from the Inspector
        if (fontAsset != null)
        {
            textMesh.font = fontAsset;
        }
        else
        {
            Debug.LogWarning("No TMP Font Asset assigned. Using default font.");
        }
        
        var rectTransform = textObj.GetComponent<RectTransform>();
        rectTransform.anchorMin = new Vector2(0, 0);
        rectTransform.anchorMax = new Vector2(1, 1);
        rectTransform.sizeDelta = Vector2.zero;
        rectTransform.localPosition = Vector3.zero;
        rectTransform.offsetMin = new Vector2(20, 40);
        rectTransform.offsetMax = new Vector2(-20, -20);
        
        canvasObj.transform.localPosition = new Vector3(0, 0.5f, 1);
        canvasObj.transform.localRotation = Quaternion.Euler(0, 180, 180);
        canvasObj.transform.localScale = new Vector3(0.003f, 0.003f, 0.003f);

        overlayTexture = new Texture_t
        {
            handle = renderTexture.GetNativeTexturePtr(),
            eType = ETextureType.DirectX,
            eColorSpace = EColorSpace.Auto
        };

        var transformMatrix = new HmdMatrix34_t();
        var pos = position;
        var rot = Quaternion.Euler(rotation);

        transformMatrix.m0 = (1f - 2f * (rot.y * rot.y + rot.z * rot.z));
        transformMatrix.m1 = (2f * (rot.x * rot.y - rot.z * rot.w));
        transformMatrix.m2 = (2f * (rot.x * rot.z + rot.y * rot.w));
        transformMatrix.m3 = pos.x;

        transformMatrix.m4 = (2f * (rot.x * rot.y + rot.z * rot.w));
        transformMatrix.m5 = (1f - 2f * (rot.x * rot.x + rot.z * rot.z));
        transformMatrix.m6 = (2f * (rot.y * rot.z - rot.x * rot.w));
        transformMatrix.m7 = pos.y;

        transformMatrix.m8 = (2f * (rot.x * rot.z - rot.y * rot.w));
        transformMatrix.m9 = (2f * (rot.y * rot.z + rot.x * rot.w));
        transformMatrix.m10 = (1f - 2f * (rot.x * rot.x + rot.y * rot.y));
        transformMatrix.m11 = pos.z;

        OpenVR.Overlay.SetOverlayTransformAbsolute(overlayHandle, ETrackingUniverseOrigin.TrackingUniverseStanding, ref transformMatrix);
        OpenVR.Overlay.SetOverlayWidthInMeters(overlayHandle, width);
        OpenVR.Overlay.ShowOverlay(overlayHandle);
    }

    private void ConnectToPythonServer()
    {
        try
        {
            client = new TcpClient(host, port);
            stream = client.GetStream();
            reader = new StreamReader(stream, Encoding.UTF8);
            Debug.Log("Connected to Python server.");
            
            keepReading = true;
            receiveThread = new Thread(new ThreadStart(ReceiveData));
            receiveThread.Start();
        }
        catch (Exception e)
        {
            Debug.LogError("Socket error: " + e.Message);
        }
    }

    void ReceiveData()
    {
        try
        {
            while (keepReading && stream != null)
            {
                string completeMessage = reader.ReadLine();
                if (completeMessage != null)
                {
                    Debug.Log($"Complete message received: {completeMessage}");
                    lock (lockObject)
                    {
                        pendingMessages.Enqueue(completeMessage);
                    }
                    needsRender = true; // Trigger render for new message
                }
                else
                {
                    Debug.LogWarning("Received null message. Server might have closed the connection.");
                    ReconnectToPythonServer();
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Receive error: {e.Message}");
            if (keepReading)
            {
                ReconnectToPythonServer();
            }
        }
    }

    private void UpdateDisplayText(string newText)
    {
        if (textMesh == null) return;

        if (!string.IsNullOrEmpty(newText))
        {
            Debug.Log($"Processing text for display: {newText}");
            lastSentences.Enqueue(new TimedMessage(newText));

            while (lastSentences.Count > MAX_SENTENCES)
            {
                lastSentences.Dequeue();
            }
        }

        // Flag to update render after adding new text
        needsRender = true;
    }

    void Update()
    {
        if (overlayHandle != OpenVR.k_ulOverlayHandleInvalid)
        {
            lock (lockObject)
            {
                if (pendingMessages.Count > 0)
                {
                    Debug.Log($"Processing {pendingMessages.Count} pending messages");
                    while (pendingMessages.Count > 0)
                    {
                        string message = pendingMessages.Dequeue();
                        UpdateDisplayText(message);
                    }
                }
            }

            // Only render if needed
            if (needsRender)
            {
                UpdateOverlayText();
                needsRender = false;
            }
        }
    }

    /// <summary>
    /// Updates the overlay text and renders the camera.
    /// </summary>
    private void UpdateOverlayText()
    {
        // Build final text with current alpha values
        StringBuilder displayBuilder = new StringBuilder();
        var currentTime = Time.time;
        var messages = lastSentences.ToArray();
        bool hasVisibleMessages = false;

        foreach (var message in messages)
        {
            if (message.Alpha > 0)
            {
                string coloredText = $"<alpha=#{Mathf.FloorToInt(message.Alpha * 255):X2}>{message.Text}";
                displayBuilder.AppendLine(coloredText);
                hasVisibleMessages = true;
            }
        }

        string finalText = displayBuilder.ToString();
        Debug.Log($"Final display text:\n{finalText}");
        
        textMesh.text = finalText;
        textMesh.ForceMeshUpdate(true);

        if (hasVisibleMessages)
        {
            textCamera.Render();
            if (overlayHandle != OpenVR.k_ulOverlayHandleInvalid && overlayTexture.handle != IntPtr.Zero)
            {
                try
                {
                    overlayTexture.handle = renderTexture.GetNativeTexturePtr();
                    OpenVR.Overlay.SetOverlayTexture(overlayHandle, ref overlayTexture);
                }
                catch (Exception e)
                {
                    Debug.LogError($"Error updating overlay texture: {e.Message}");
                }
            }
        }
    }

    /// <summary>
    /// Coroutine to continuously update message fading smoothly.
    /// </summary>
    /// <returns></returns>
    private System.Collections.IEnumerator FadeMessagesRoutine()
    {
        while (true)
        {
            bool requiresRender = false;

            // Update message alphas based on elapsed time
            var currentTime = Time.time;
            var messages = lastSentences.ToArray();

            foreach (var message in messages)
            {
                float messageAge = currentTime - message.TimeReceived;

                if (messageAge > messageFadeStartTime)
                {
                    float fadeProgress = (messageAge - messageFadeStartTime) / messageFadeDuration;
                    float newAlpha = Mathf.Clamp01(1.0f - fadeProgress);

                    if (Mathf.Abs(newAlpha - message.Alpha) > 0.01f)
                    {
                        message.Alpha = newAlpha;
                        requiresRender = true;
                    }
                }
            }

            // Remove messages that have fully faded
            while (lastSentences.Count > 0 && lastSentences.Peek().Alpha <= 0)
            {
                lastSentences.Dequeue();
                requiresRender = true;
            }

            if (requiresRender)
            {
                needsRender = true;
            }

            yield return new WaitForSeconds(fadeUpdateInterval); // Wait for the specified interval
        }
    }

    void OnDestroy()
    {
        try
        {
            keepReading = false;
            
            if (receiveThread != null && receiveThread.IsAlive)
            {
                try
                {
                    receiveThread.Join(1000);
                    if (receiveThread.IsAlive)
                    {
                        receiveThread.Abort();
                    }
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"Error closing receive thread: {e.Message}");
                }
            }

            if (stream != null)
            {
                try { stream.Close(); } catch { }
                stream = null;
            }
            
            if (client != null)
            {
                try { client.Close(); } catch { }
                client = null;
            }
            
            if (overlayHandle != OpenVR.k_ulOverlayHandleInvalid && OpenVR.Overlay != null)
            {
                try { OpenVR.Overlay.DestroyOverlay(overlayHandle); } catch { }
            }
            
            if (renderTexture != null)
            {
                try { renderTexture.Release(); } catch { }
                renderTexture = null;
            }

            // Destroy camera and canvas to free resources
            if (textCamera != null)
            {
                Destroy(textCamera.gameObject);
                textCamera = null;
            }

            // Stop the fade coroutine
            if (fadeCoroutine != null)
            {
                StopCoroutine(fadeCoroutine);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error in OnDestroy: {e.Message}");
        }
    }

    private void ReconnectToPythonServer()
    {
        try
        {
            if (client != null)
            {
                client.Close();
                client = null;
            }
            if (stream != null)
            {
                stream.Close();
                stream = null;
            }

            Thread.Sleep(1000);
            ConnectToPythonServer();
        }
        catch (Exception e)
        {
            Debug.LogError($"Reconnection error: {e.Message}");
        }
    }
}
