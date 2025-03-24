# Acoustic Multi-target Tracking (AMT)

This code implements a paper called AMT: Acoustic Multi-target Tracking with Smartphone MIMO System, which does Acoustic target tracking using multiple microphones and speakers.

## Bug Analysis: Position Shift

A constant position shift bug has been observed where sometimes the targets are off by a constant shift from their actual location. To help diagnose this issue, diagnostic images are saved to the `amt_debug_images` directory when running the simulation.

### Debugging Strategy

When analyzing the debug images, look for:

1. **Error Maps**: These visualize the multilateration error landscape. Look for:
   - Multiple local minima that might confuse the position estimation
   - Asymmetric error patterns that might bias the estimation in a particular direction
   - Different optimal positions across different Z-heights

2. **Per-Axis Error Analysis**: Look at the tracking_errors.png file to see:
   - If there's a consistent bias in a particular direction (X, Y, or Z)
   - If the shift is truly constant or if it changes over time
   - In particular, check if any axis has a consistent non-zero mean error

3. **Correlation Data**: In the correlation visualization images, pay attention to:
   - The difference between expected and detected peaks
   - Whether the direct path delay estimation is accurate
   - Any systematic errors in the time-of-flight measurements

4. **Ellipse Analysis**: In the ellipses visualization, check:
   - If the ellipses are correctly positioned based on speakers and mics
   - If the target true position is at the intersection of multiple ellipses
   - If there are systematic biases in the ellipse positions

### Potential Causes

The constant shift could be caused by several factors:

1. **Calibration Error**: Incorrect speaker or microphone positions
2. **Speed of Sound Error**: Incorrect speed of sound parameter
3. **Multilateration Algorithm Issues**: Local minima or biased error functions
4. **Direct Path Delay Issues**: Incorrect calculation of the direct path delays
5. **Systematic Measurement Bias**: Consistent error in time-of-flight or correlation peak detection

### Solutions to Try

After analyzing the debug images, consider:

1. If the shift is truly constant, adding a calibration offset to compensate
2. Refining the multilateration algorithm to be more robust to multiple minima
3. Improving the correlation peak detection with more precise filtering
4. Adjusting the Kalman filter parameters to be more/less reliant on measurements
5. Adding more rigorous validation of multilateration results before accepting them