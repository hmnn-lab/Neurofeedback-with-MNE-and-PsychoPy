using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using LSL;
using TMPro;
using System.Collections.Generic;

public class Sample_cubegame : MonoBehaviour
{
    public GameObject cube;
    public TextMeshProUGUI feedbackText;
    public TextMeshProUGUI instructionText;
    public TextMeshProUGUI levelText;
    public TextMeshProUGUI scoreText;
    public Image rulerFill;
    public TextMeshProUGUI rulerMinLabel;
    public TextMeshProUGUI rulerMaxLabel;
    public Button playButton;
    public Button quitButton; // New Quit button
    public GameObject endScreen;
    public GameObject starDisplay;
    public GameObject starPrefab;
    public GameObject loadingPanel;
    public TextMeshProUGUI loadingText;
    public GameObject preprocessingPanel;
    public TextMeshProUGUI preprocessingText;

    private StreamInlet inlet;
    private float[] sample;
    private double lastSampleTime;
    private float feedbackValue = 0f;
    private float previousFeedbackValue = 0f;

    private float upperBound = 10f;
    private float startingUpperBound = 10f;
    private int level = 1;
    private int score = 0;
    private bool isGameRunning = false;
    private bool hasCrossed75 = false;
    private List<GameObject> stars = new List<GameObject>();

    private float minY = 1f;
    private float maxY = 21f;
    private readonly float fixedX = 0f;
    private readonly float fixedZ = 0f;

    private float initialCameraZ = -10f;
    private float cameraZoomFactor = 1.5f;

    void Start()
    {
        instructionText.text = "Press Play to start!";
        feedbackText.text = "Feedback: 0";
        levelText.text = "Level: 1";
        scoreText.text = "Score: 0";
        endScreen.SetActive(false);
        loadingPanel.SetActive(false);
        preprocessingPanel.SetActive(false);

        // Verify panel references
        if (preprocessingPanel == null)
        {
            UnityEngine.Debug.LogError("PreprocessingPanel is not assigned!");
        }
        if (loadingPanel == null)
        {
            UnityEngine.Debug.LogError("LoadingPanel is not assigned!");
        }

        if (rulerFill != null)
        {
            rulerFill.fillAmount = 0f;
        }
        if (rulerMinLabel != null)
        {
            rulerMinLabel.text = "0";
        }
        if (rulerMaxLabel != null)
        {
            rulerMaxLabel.text = "10";
        }

        playButton.onClick.AddListener(StartEEGProcessing);
        playButton.gameObject.SetActive(true);

        // Setup Quit button
        if (quitButton != null)
        {
            quitButton.onClick.AddListener(QuitApplication);
            quitButton.gameObject.SetActive(true); // Always visible
        }
        else
        {
            UnityEngine.Debug.LogWarning("Quit button is not assigned! Please assign it in the inspector.");
        }

        if (cube != null && cube.GetComponent<Renderer>() != null)
        {
            cube.GetComponent<Renderer>().material.color = Color.white;
            cube.transform.localScale = Vector3.one;
            UnityEngine.Debug.Log("Cube initialized with white color and scale (1,1,1).");
        }
        else
        {
            UnityEngine.Debug.LogError("Cube or Renderer component is missing!");
        }

        if (cube != null)
        {
            // Position cube at bottom center of screen
            PositionCubeAtBottomCenter();
            UnityEngine.Debug.Log($"Cube initial position: {cube.transform.position}");
        }
        else
        {
            UnityEngine.Debug.LogError("Cube reference is missing!");
        }

        if (starPrefab == null)
        {
            UnityEngine.Debug.LogError("Star Prefab not assigned!");
        }

        if (cube != null && cube.GetComponent<Rigidbody>() != null)
        {
            UnityEngine.Debug.LogWarning("Cube has a Rigidbody. Setting to kinematic to allow scripted movement.");
            cube.GetComponent<Rigidbody>().isKinematic = true;
        }

        if (preprocessingPanel != null && preprocessingPanel.GetComponentInChildren<Spinner>() != null)
        {
            UnityEngine.Debug.Log("Spinner initialized on preprocessing panel.");
        }
        else
        {
            UnityEngine.Debug.LogWarning("Spinner component missing on preprocessing panel or panel not assigned!");
        }

        Camera activeCamera = Camera.main;
        if (activeCamera != null)
        {
            var cinemachine = activeCamera.GetComponent("CinemachineBrain");
            if (cinemachine != null)
            {
                UnityEngine.Debug.LogWarning("CinemachineBrain found on Main Camera. Disabling to prevent override.");
                (cinemachine as MonoBehaviour).enabled = false;
            }

            var components = activeCamera.GetComponents<Component>();
            string componentList = "Main Camera components:\n";
            foreach (var comp in components)
            {
                componentList += $"- {comp.GetType().Name}\n";
            }
            UnityEngine.Debug.Log(componentList);

            if (activeCamera.transform.parent != null)
            {
                UnityEngine.Debug.LogWarning($"Main Camera is parented to {activeCamera.transform.parent.name}. World position: {activeCamera.transform.position}, Local position: {activeCamera.transform.localPosition}");
            }

            activeCamera.transform.position = new Vector3(0f, 1f, initialCameraZ);
            activeCamera.transform.rotation = Quaternion.Euler(0f, 0f, 0f);
            UnityEngine.Debug.Log($"Main Camera initialized to position: {activeCamera.transform.position}, rotation: {activeCamera.transform.rotation.eulerAngles}, forward: {activeCamera.transform.forward}");
        }
        else
        {
            UnityEngine.Debug.LogError("No Main Camera found in the scene!");
        }
    }

    void StartEEGProcessing()
    {
        playButton.interactable = false;
        // Keep quit button visible during gameplay
        preprocessingPanel.SetActive(true);
        preprocessingText.text = "Connecting to EEG Stream...";
        StartCoroutine(InitializeLSL());
        StartCoroutine(AnimatePreprocessingText());
    }

    IEnumerator InitializeLSL()
    {
        UnityEngine.Debug.Log("Waiting for EEG stream to initialize...");
        preprocessingText.text = "Waiting for EEG Stream...";

        int maxAttempts = 20; // Allow ~100 seconds for stream to appear
        float waitTime = 5.0f;
        for (int attempt = 1; attempt <= maxAttempts; attempt++)
        {
            UnityEngine.Debug.Log($"Attempting to resolve LSL stream 'BandPowerChange' (Attempt {attempt}/{maxAttempts})...");
            preprocessingPanel.SetActive(false);
            loadingPanel.SetActive(true);
            loadingText.text = $"Searching for EEG Stream (Attempt {attempt}/{maxAttempts})...";

            try
            {
                StreamInfo[] allStreams = LSL.LSL.resolve_streams(waitTime);
                if (allStreams.Length > 0)
                {
                    string streamList = "Available LSL streams:\n";
                    foreach (var stream in allStreams)
                    {
                        streamList += $"- Name: {stream.name()}, Type: {stream.type()}, Channels: {stream.channel_count()}\n";
                    }
                    UnityEngine.Debug.Log(streamList);
                }
                else
                {
                    UnityEngine.Debug.LogWarning("No LSL streams found.");
                }
            }
            catch (System.Exception e)
            {
                UnityEngine.Debug.LogError($"Error listing LSL streams: {e.Message}");
            }

            StreamInfo[] results = null;
            bool streamResolved = false;

            try
            {
                results = LSL.LSL.resolve_stream("name", "BandPowerChange", 1, waitTime);
                streamResolved = results.Length > 0;
            }
            catch (System.Exception e)
            {
                UnityEngine.Debug.LogError($"Error resolving LSL stream: {e.Message}");
            }

            if (streamResolved)
            {
                inlet = new StreamInlet(results[0]);
                inlet.open_stream();
                sample = new float[results[0].channel_count()];
                UnityEngine.Debug.Log($"LSL stream connected successfully. Channel count: {results[0].channel_count()}, Stream Name: {results[0].name()}, Type: {results[0].type()}");
                isGameRunning = true;
                playButton.gameObject.SetActive(false);
                instructionText.text = "Raise the cube!";
                // Explicitly hide both panels
                preprocessingPanel.SetActive(false);
                loadingPanel.SetActive(false);
                yield break;
            }
            else
            {
                UnityEngine.Debug.LogWarning($"No LSL stream 'BandPowerChange' found on attempt {attempt}. Retrying...");
                yield return new WaitForSeconds(1.0f);
            }
        }

        HandleLSLFailure();
    }

    private IEnumerator AnimatePreprocessingText()
    {
        while (!isGameRunning)
        {
            preprocessingText.text = "Waiting for EEG Stream" + new string('.', (int)(Time.time % 4));
            yield return new WaitForSeconds(0.5f);
        }
    }

    void HandleLSLFailure()
    {
        UnityEngine.Debug.LogError("Failed to find LSL stream 'BandPowerChange' after multiple attempts.");
        isGameRunning = false;
        preprocessingPanel.SetActive(false);
        loadingPanel.SetActive(false);
        instructionText.text = "Error: Could not connect to EEG stream. Ensure 'Start Streaming' is clicked in EEG UI.";
        playButton.gameObject.SetActive(true);
        playButton.interactable = true;
        // Quit button remains visible
        EndGame();
    }

    void Update()
    {
        if (!isGameRunning)
        {
            // Allow quitting even when game is not running
            if (Input.GetKeyDown(KeyCode.Escape))
            {
                QuitApplication();
            }
            return;
        }

        Camera activeCamera = Camera.main;
        if (activeCamera != null)
        {
            float targetCameraZ = initialCameraZ * Mathf.Pow(cameraZoomFactor, level - 1);
            activeCamera.transform.position = new Vector3(0f, 1f, targetCameraZ);
            activeCamera.transform.rotation = Quaternion.Euler(0f, 0f, 0f);
            UnityEngine.Debug.Log($"Main Camera runtime position: {activeCamera.transform.position}, rotation: {activeCamera.transform.rotation.eulerAngles}, forward: {activeCamera.transform.forward}");
        }

        if (cube != null)
        {
            UnityEngine.Debug.Log($"Cube position: {cube.transform.position}, scale: {cube.transform.localScale}");
        }
        else
        {
            UnityEngine.Debug.LogError("Cube reference is missing in Update!");
            return;
        }

        if (inlet != null)
        {
            try
            {
                double timestamp = inlet.pull_sample(sample, 0.01);
                if (timestamp > 0)
                {
                    feedbackValue = sample[0];
                    lastSampleTime = timestamp;
                    UnityEngine.Debug.Log($"Received LSL sample: {feedbackValue:F2} (Timestamp: {timestamp:F6}, Sample[0]: {sample[0]:F2})");
                }
                else
                {
                    UnityEngine.Debug.LogWarning("No new LSL sample received.");
                }

                if (inlet.info() == null)
                {
                    UnityEngine.Debug.LogError("LSL inlet is invalid or stream was closed.");
                    isGameRunning = false;
                    EndGame();
                    return;
                }
            }
            catch (System.Exception e)
            {
                UnityEngine.Debug.LogError($"Error pulling LSL sample: {e.Message}");
            }
        }
        else
        {
            UnityEngine.Debug.LogWarning("LSL inlet is null. Cannot pull samples.");
            isGameRunning = false;
            EndGame();
            return;
        }

        float gameUpperBound = 10f * Mathf.Pow(2, level - 1);
        float feedbackFraction = Mathf.Clamp01(feedbackValue / upperBound);

        float upperThreshold = 0.75f * gameUpperBound;
        float lowerThreshold = 0.25f * gameUpperBound;
        UnityEngine.Debug.Log($"Feedback: {feedbackValue:F2}, FeedbackFraction: {feedbackFraction:F2}, UpperBound: {upperBound:F2}, GameUpperBound: {gameUpperBound:F2}, UpperThreshold: {upperThreshold:F2}, LowerThreshold: {lowerThreshold:F2}, Level: {level}");

        float feedbackDelta = feedbackValue - previousFeedbackValue;
        if (Mathf.Abs(feedbackDelta) > 0.01f)
        {
            if (feedbackDelta > 0)
            {
                score += 2;
            }
            else
            {
                score -= 2;
            }
            score = Mathf.Max(0, score);
            scoreText.text = $"Score: {score}";
            UnityEngine.Debug.Log($"Score updated: {score}, Feedback Delta: {feedbackDelta:F2}");
        }
        previousFeedbackValue = feedbackValue;

        if (cube != null)
        {
            float targetY = Mathf.Lerp(minY, maxY, feedbackFraction);
            cube.transform.position = Vector3.Lerp(cube.transform.position, new Vector3(cube.transform.position.x, targetY, fixedZ), Time.deltaTime * 5f);
        }

        feedbackText.text = $"Feedback: {feedbackValue:F2}";
        if (rulerFill != null)
        {
            rulerFill.fillAmount = feedbackFraction;
        }

        foreach (var star in stars)
        {
            if (star != null)
            {
                RectTransform starRect = star.GetComponent<RectTransform>();
                if (starRect != null)
                {
                    float baseScale = 1f;
                    float pulseAmplitude = 0.08f;
                    float scoreFactor = Mathf.Clamp01(score / 100f);
                    float pulse = Mathf.Sin(Time.time * 2f + star.GetInstanceID() * 0.1f) * pulseAmplitude * (0.5f + scoreFactor * 0.5f);
                    float scale = baseScale + pulse;
                    starRect.localScale = new Vector3(scale, scale, 1f);
                }
            }
        }

        if (!hasCrossed75 && feedbackValue >= upperThreshold)
        {
            LevelUp();
            hasCrossed75 = true;
            UnityEngine.Debug.Log($"Feedback {feedbackValue:F2} crossed 75% of game boundary {upperThreshold:F2}");
        }
        else if (hasCrossed75 && feedbackValue < upperThreshold * 0.9f)
        {
            hasCrossed75 = false;
            UnityEngine.Debug.Log($"Feedback {feedbackValue:F2} dropped below reset threshold, ready to re-cross");
        }

        if (feedbackValue <= lowerThreshold && level > 1)
        {
            LevelDown();
            UnityEngine.Debug.Log($"Feedback {feedbackValue:F2} dropped below 25% of game boundary {lowerThreshold:F2}, leveling down");
        }

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            EndGame();
        }
    }

    void LevelUp()
    {
        level++;
        score += 10;
        upperBound *= 2;
        startingUpperBound = upperBound;
        float gameUpperBound = 10f * Mathf.Pow(2, level - 1);

        if (rulerMaxLabel != null)
        {
            rulerMaxLabel.text = gameUpperBound.ToString("F0");
        }

        if (cube != null && cube.GetComponent<Renderer>() != null)
        {
            Color newColor;
            if (level == 1)
            {
                newColor = new Color(1.0f, 0.2f, 0.2f);
                UnityEngine.Debug.Log($"Cube color changed to Bright red: {newColor} at level {level}");
            }
            else if (level == 2)
            {
                newColor = new Color(0.86f, 0.08f, 0.24f);
                UnityEngine.Debug.Log($"Cube color changed to Chilli red: {newColor} at level {level}");
            }
            else
            {
                newColor = new Color(0.55f, 0.0f, 0.0f);
                UnityEngine.Debug.Log($"Cube color changed to Dark red: {newColor} at level {level}");
            }
            cube.GetComponent<Renderer>().material.color = newColor;
        }
        else
        {
            UnityEngine.Debug.LogError("Cannot change cube color: Cube or Renderer missing!");
        }

        if (starPrefab != null && starDisplay != null)
        {
            GameObject newStar = Instantiate(starPrefab, starDisplay.transform);
            RectTransform starRect = newStar.GetComponent<RectTransform>();
            starRect.anchoredPosition = new Vector2((stars.Count * 60f), 0);
            starRect.sizeDelta = new Vector2(40f, 40f);
            starRect.localScale = Vector3.one;
            stars.Add(newStar);
            UnityEngine.Debug.Log($"Added star {stars.Count} for level {level}, size: {starRect.sizeDelta}, scale: {starRect.localScale}");
        }

        levelText.text = $"Level: {level}";
        scoreText.text = $"Score: {score}";
        UnityEngine.Debug.Log($"Leveled up to {level}, new upper bound: {upperBound}, starting upper bound: {startingUpperBound}");
    }

    void LevelDown()
    {
        if (level <= 1)
        {
            UnityEngine.Debug.Log("Cannot level down: Already at level 1");
            return;
        }

        level--;
        score = Mathf.Max(0, score - 10);
        upperBound /= 2;
        startingUpperBound = upperBound;
        float gameUpperBound = 10f * Mathf.Pow(2, level - 1);

        if (rulerMaxLabel != null)
        {
            rulerMaxLabel.text = gameUpperBound.ToString("F0");
        }

        if (cube != null && cube.GetComponent<Renderer>() != null)
        {
            Color newColor;
            if (level == 1)
            {
                newColor = new Color(1.0f, 0.2f, 0.2f);
                UnityEngine.Debug.Log($"Cube color changed to Bright red: {newColor} at level {level}");
            }
            else if (level == 2)
            {
                newColor = new Color(0.86f, 0.08f, 0.24f);
                UnityEngine.Debug.Log($"Cube color changed to Chilli red: {newColor} at level {level}");
            }
            else
            {
                newColor = new Color(0.55f, 0.0f, 0.0f);
                UnityEngine.Debug.Log($"Cube color changed to Dark red: {newColor} at level {level}");
            }
            cube.GetComponent<Renderer>().material.color = newColor;
        }
        else
        {
            UnityEngine.Debug.LogError("Cannot change cube color: Cube or Renderer missing!");
        }

        if (stars.Count > 0)
        {
            GameObject starToDestroy = stars[stars.Count - 1];
            stars.RemoveAt(stars.Count - 1);
            if (starToDestroy != null)
            {
                Destroy(starToDestroy);
                UnityEngine.Debug.Log($"Destroyed star, remaining stars: {stars.Count}");
            }
        }

        levelText.text = $"Level: {level}";
        scoreText.text = $"Score: {score}";
        UnityEngine.Debug.Log($"Leveled down to {level}, new upper bound: {upperBound}, starting upper bound: {startingUpperBound}");
    }

    void EndGame()
    {
        isGameRunning = false;
        preprocessingPanel.SetActive(false);
        endScreen.SetActive(true);
        instructionText.text = "Game Over! Press Play to restart or Quit to exit.";
        playButton.gameObject.SetActive(true);
        playButton.interactable = true;

        // Quit button remains visible (no need to show/hide)

        if (inlet != null)
        {
            inlet.close_stream();
        }
    }

    void PositionCubeAtBottomCenter()
    {
        Camera activeCamera = Camera.main;
        if (activeCamera != null && cube != null)
        {
            // Get screen dimensions
            float screenWidth = Screen.width;
            float screenHeight = Screen.height;

            // Define bottom center position in screen coordinates
            // Use a small offset from the very bottom (e.g., 10% of screen height)
            Vector3 screenPosition = new Vector3(screenWidth * 0.5f, screenHeight * 0.1f, 0f);

            // Convert screen position to world position
            // Use the cube's current Z position or a default distance from camera
            float distanceFromCamera = Mathf.Abs(activeCamera.transform.position.z - fixedZ);
            screenPosition.z = distanceFromCamera;

            Vector3 worldPosition = activeCamera.ScreenToWorldPoint(screenPosition);

            // Keep the fixed Z coordinate
            worldPosition.z = fixedZ;

            // Set the cube position
            cube.transform.position = worldPosition;

            // Update minY to this new bottom position for consistent movement range
            minY = worldPosition.y;

            // Calculate maxY based on screen height (e.g., 90% of screen height)
            Vector3 topScreenPosition = new Vector3(screenWidth * 0.5f, screenHeight * 0.9f, distanceFromCamera);
            Vector3 topWorldPosition = activeCamera.ScreenToWorldPoint(topScreenPosition);
            maxY = topWorldPosition.y;

            UnityEngine.Debug.Log($"Cube positioned at bottom center: {worldPosition}, minY: {minY}, maxY: {maxY}");
        }
        else
        {
            UnityEngine.Debug.LogError("Camera or Cube reference missing for positioning!");
        }
    }

    void QuitApplication()
    {
        UnityEngine.Debug.Log("Quitting application...");

        // Close LSL stream if still open
        if (inlet != null)
        {
            inlet.close_stream();
        }

        // Clean up stars
        foreach (var star in stars)
        {
            if (star != null) Destroy(star);
        }
        stars.Clear();

        // Quit the application
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
    }

    void OnDestroy()
    {
        if (inlet != null)
        {
            inlet.close_stream();
        }

        foreach (var star in stars)
        {
            if (star != null) Destroy(star);
        }
        stars.Clear();
    }
}