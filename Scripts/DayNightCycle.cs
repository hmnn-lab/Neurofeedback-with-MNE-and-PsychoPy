using UnityEngine;

public class DayNightCycle : MonoBehaviour
{
    [Tooltip("Duration of a full day-night cycle in seconds")]
    public float dayDuration = 60f; // Default to 60 seconds for a full cycle

    private float rotationSpeed;

    void Start()
    {
        // Calculate the rotation speed (360 degrees per full cycle)
        rotationSpeed = 360f / dayDuration;
    }

    void Update()
    {
        // Rotate the light around the X-axis to simulate the sun moving
        transform.Rotate(Vector3.right, rotationSpeed * Time.deltaTime);
    }
}
