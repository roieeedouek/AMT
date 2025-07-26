# AMT+: Acoustic Multi-Target Tracking With Smartphone MIMO System

**Authors:** Penghao Wang, Ruobing Jiang, Jingyang Hu (Member, IEEE), Yanmin Zhu, Hongbo Jiang (Senior Member, IEEE), Minglu Li (Fellow, IEEE), and Chao Liu (Member, IEEE)

**Published:** IEEE Transactions on Mobile Computing, Vol. 23, No. 12, December 2024

## Abstract

Acoustic target tracking has shown great advantages for device-free human-machine interaction over vision/RF-based mechanisms. However, existing approaches for portable devices solely track a single target, incapable of the ubiquitous and highly challenging multi-target situations such as double-hand multimedia controlling and multi-player gaming. In this paper, we propose AMT+, a pioneering smartphone MIMO system to achieve centimeter-level multi-target tracking. The challenge of multi-target occlusion is effectively addressed by employing multiple speaker-microphone pairs. However, the unique challenge raised by MIMO is the superposition of multi-source signals due to the cross-correlation among speakers. Initially, we tackle this challenge by designing a weak cross-correlation signal to reduce interference passively. In AMT+, we've further integrated self-interference cancellation for active minimize interference. The most distinguishing advantage of AMT+ lies in the elimination of the raised multipath effect, which is commonly ignored in previous work by hastily assuming targets as particles. AMT+ employs Doppler filtering over delay subtraction for echo suppression. Further, by non-particle target reflections modeling results, we introduce a distance-projection-based method for continuous target identification and tracking. Implemented on commercial smartphones, AMT+ achieves on average 0.54 cm, 1.37 cm, and 2.13 cm errors for single, double, and triple target tracking respectively, and on average 97.0% classification accuracy for 14 controlling gestures.

**Index Terms:** Multi-target tracking, acoustic MIMO system, primary echo detection, multipath effect elimination.

## I. Introduction

Device-free target tracking and gesture recognition have attracted increasing attention for the convenience of human-machine interaction. Target tracking based on acoustic signals is preferred over vision-based solutions with lighting requirements and RF-based ones with coarse-grained resolution and lower accuracy. However, existing acoustic approaches based on portable devices, e.g., smartphones, only track a single target, which therefore exclusively forbids the movement of any other objects except the aimed single target.

As a result, multi-target tracking becomes a crucial necessity to improve the convenience and applicability of acoustic tracking. Promising applications of acoustic multi-target tracking include:

- **Complex double-hand operations** on pictures (remote zoom in/out) and videos (play/pause/fast-forward/backward)
- **Multi-player gaming** where acoustic gesture recognition extends the interaction space beyond the limited screen to avoid screen occlusions and mutual interference among players

Previous acoustic single-target tracking approaches applicable on portable devices apply Doppler Frequency Shift (DFS), phase offset, Channel State Information (CSI), or Time of Flight (ToF), exhibit three main disadvantages:

1. **DFS/phase/CSI based approaches** cumulatively measure target distance change instead of directly ranging absolute distance. Serious accumulative errors are thus induced, and the unknown target initial position needs to be determined.

2. **DFS/phase-based approaches** measure a combined impact from multiple echoes on frequency/phase and thus cannot reduce the multipath effect.

3. **ToF-based approaches** directly range target absolute distance but still cannot eliminate the multipath effect caused by single/multiple targets.

## II. Related Work

### A. Vision Based Tracking

Vision-based approaches apply real-time image recognition to capture targets through camera shooting, having stringent requirements on the environment lighting. For instance, uSensAR, a smartphone camera-based 3D gesture analysis framework, facilitates hand positioning and gesture identification under favorable lighting and field of vision. However, the tracking performance of these approaches is significantly limited by lens angle and resolution.

### B. RF Based Tracking

RF-based approaches only identify coarse motions since RF signals travel so fast as light speed, which is near one million times faster than sound. For instance, the Google Pixel 4 once utilized the integrated Soli radar to transmit 60GHz RF signals, leveraging Doppler shifts for simple gesture recognition. However, due to limitations in cost, energy consumption, and accuracy, this feature was discontinued.

### C. Acoustic Approaches

Acoustic approaches employ much slower sound waves to improve tracking accuracy. A target is located by measuring its impacts on signal metrics of the reflected echoes, e.g., Doppler Frequency Shift (DFS), phase offset, Channel State Information (CSI), and Time of Flight (ToF). According to the employed metrics, previous acoustic approaches can further be classified as follows:

#### 1) Acoustic Tracking Using Frequency Shift or Phase Offset
Target distance change is achieved by accumulating the DFS or phase offset caused by target movements. Disadvantages are two-fold:
- Multipath echoes raised by single target are ignored by hastily assuming targets as particles
- Frequency/phase shift essentially indicates distance change instead of absolute distance

#### 2) Acoustic Tracking Using CSI
To separate multipath echoes unrealizable for frequency/phase shift based approaches, channel-oriented CSI has been proposed. Time-domain channel estimation is employed to select a representative echo path to perform phase change detection.

#### 3) Acoustic Tracking Using ToF
ToF has widely been used to range target absolute distance at device end to avoid accumulative error caused by displacement integration. Target absolute distance is estimated in real-time by recording the round-trip duration of the acoustic echo from signal source via target to receiver.

## III. Overview

### A. Motivation

Existing acoustic target tracking approaches reveal three significant problems:

**Problem 1. Impractical Particle Assumption:** All existing acoustic target tracking systems impractically assume each target as a particle, ignoring the multipath effect raised by the target itself.

**Problem 2. Incapable for Multi-target Tracking without MIMO:** Phase/frequency/CSI-based methods are cognitively incapable for multi-target tracking while ToF-based approaches without MIMO neither involve sufficient clues for multi-target tracking.

**Problem 3. Target confusion during multi-target tracking:** During target tracking, the movement trajectories of different targets may cross at a certain point in time.

We first introduce existing ToF-based positioning schemes, which locate a particle target by intersecting ranging ellipses.

**Definition 1. (Acoustic ranging ellipse):** An acoustic ranging ellipse concerning a target A is focused on a pair of speaker S and microphone R, denoted by ΘA(S, R). The major axis length of the ranging ellipse equals to the path length of the echo from S via A to R.

### B. Basic Idea

To solve this, we propose a MIMO system for non-particle target to address both the problems identified above.

**Definition 2. (Indicating Point and Primary Echo):** The indicating point of a moving non-particle target is the fastest moving point of the target among all the points on the target surface, e.g., index fingertip of a hand. The primary echo ρA(S, R) of a target A concerning a given speaker-mic pair (S, R) is the echo reflected by the indicating point of the target.

**Definition 3. (Target Distance Projection):** Multiple echoes from non-particle targets result in a unique peak arrangement in pulse compression results. This arrangement represents the distance projection of the target and includes detailed shape information.

Our basic idea is to distinguish the primary echo among all the multipath echoes of each target per speaker-mic pair and perform multi-lateration locating with at least 3 ranging ellipses focused on distinct speaker-mic pairs.

The basic idea can be implemented by following 5 key steps:

1. **Source separation:** Each microphone separates the received mixed signals from different speakers
2. **Receiver synchronization:** Calculate the sending time ts to achieve time synchronization
3. **Interference elimination:** Use higher-order Moving Target Indicator (MTI) filter and self-adaptive filter
4. **Primary echo detection:** Detect the primary echo among all multipath echoes
5. **Multi-target identification and tracking:** Perform multi-lateration locating and tracking for each target

### C. Challenges

There exist three significant challenges for acoustic multi-target tracking:

**Challenge 1. Indistinguishable source due to cross-correlation among speakers:** A real-world microphone continuously receives a superposition of both direct copies and multipath echoes from all the speakers.

**Challenge 2. Inaccurate arrival time due to multipath effect:** It is difficult to detect either the direct arrived copy or reflected copies due to multipath effects.

**Challenge 3. Temporally continuous multi-target identification and tracking:** A non-particle target generates multiple echoes along different paths, and using one path to represent the target overlooks differences between targets.

## IV. Signal Foundation

The innovative sending scheme of AMT+ continuously sends the periodic ZC sequence as the baseband signal to completely eliminate Inter-Channel Interference (ICI) and Inter-Symbol Interference (ISI).

### A. Signal Design

The continuously sent ZC sequence, consisting of periodic root sequences, is constant-amplitude and zero auto-correlated, which can be expressed by:

```
z[n] = {
  e^(-j2π u/Nzc * (n(n+1)/2 + qn)), for Nzc odd
  e^(-j2π u/Nzc * (n²/2 + qn)),       for Nzc even
}
```

where Nzc is the length of root ZC sequence, u is a positive integer less than Nzc and gcd(u, Nzc)=1, and q is any integer.

The advantages of ZC sequence are two-fold:
- **Strong Auto-correlation:** The auto-correlation between a root sequence and its cyclically shifted version is zero
- **Weak Cross-correlation:** Two ZC sequences with different indicators are weakly cross-correlated

### B. Modulation and Demodulation

**Modulation:** AMT+ modulates a raw complex valued ZC sequence onto the orthogonal sub-carriers within the inaudible frequency band. The modulated signal is directly achieved without interpolation and up-conversion.

**Demodulation:** AMT+ converts the received original signal yj[n] by microphone Rj to the corresponding IQ signal rj[n].

## V. Source Separation

### A. MIMO Multipath Model

The channel between a speaker and a microphone is a linear time-variant channel with Channel Impulse Response (CIR), denoted as h(d, t), varying with time t and the distance d between the reflective object and the phone.

The receipt signal r(d, t) can be expressed as the convolution:
```
r(d, t) = s(t) * h(d, t)
```

A 2 × 2 MIMO multipath model can be expressed as:
```
r1 = s1 * h11 + s2 * h21
r2 = s1 * h12 + s2 * h22
```

### B. Source Enhancement With Self-Inference Cancellation

Under the 2 × 2 MIMO multipath model, each microphone needs to differentiate and separate multi-source signals from the received mix. The enhanced signal r'1 is achieved by:
```
r'1 = r1 - s2 * ĥ21 = s1 * h11 + σ · s2 * h21
```
where σ → 0.

### C. Source Separation With Sliding-Window Correlation

With the enhanced signal r'j where signal components from other speakers are largely diminished, the target components can be separated with sliding-window correlation using the strong auto-correlation and weak cross-correlation properties of ZC sequences.

## VI. Source-Oriented Direct Signal Detection

AMT+ synchronizes receivers with signal sending time ts by detecting precise direct arrival time td. The direct signal is first identified as the max-amplitude peak in the correlation function and then td is corrected with the phase offset of the direct signal.

### A. Correlation Detection Based Direct Signal Detection

The arrival of the direct copy can be detected as the sample with the maximum amplitude peak in the correlation result:
```
n̂d,ij = arg max |Rij[n]|
         n=0 to N
```

### B. Phase Offset Based td Correction

We correct n̂d to achieve accurate td by analyzing the phase offset of the signal segment on the raw receipt signal y before demodulation, accounting for symbol/sampling time offset and frequency offset.

## VII. Non-Primary Influence and Noise Elimination

AMT+ filters out non-primary influence from both static objects and mobile side parts of targets while incorporating higher-order MTI filter and self-adaptive threshold for optimal performance.

### A. Static Influence Elimination With Consecutive Subtraction

The subtraction result of R[n] is denoted as:
```
D[n] = |R[n] - R[n + N]|
```

### B. Non-Primary Influence Elimination With High-Order MTI

The 4-order MTI filter is employed:
```
hAMT+(n) = δ(n) - 4δ(n - N) + 6δ(n - 2N) - 4δ(n - 3N) + δ(n - 4N)
```

### C. Environment Noise Elimination With Self-Adaptive Filter

We introduced a novel adaptive threshold:
```
Th = mean(D) + k · std(D)
```

## VIII. Primary Echo Detection and Multi-Target Tracking

### A. Primary Echo-Based Target Separation and Path Measurement

By detecting primary echoes from different targets, AMT+ can separate multi-target echoes and measure propagation path length. Primary echo detection is achieved by scanning D[n] for a large peak with multiple small peaks.

### B. Continuous Multi-Target Identification and Tracking by Distance Projection

We design a distance-projection-based target identification and tracking algorithm that uses the correlation between current and previous distance projections to identify targets continuously.

### C. Multi-Ellipse Localization Algorithm

Multiple ellipses may not precisely intersect at the same point due to measurement errors. AMT+ uses multiple speaker-microphone pairs to construct ellipses and determine target location through refined localization.

## IX. Performance Evaluation

The proposed AMT+ was evaluated on various hardware devices including Samsung S7, Razer Phone 2, Samsung Note10+, Xiaomi 10 Ultra, Samsung S23 Ultra, and a Google phone. 20 volunteers (15 females and 5 males aged between 22 and 42) participated in experiments.

### A. 1-D Finger Ranging

**Average ranging error:** AMT+ achieved 0.42 cm average ranging error with 90% ranging error of 1.12 cm, significantly lower than AMT and CC-ToF.

**Device compatibility:** Ranging errors across various devices showed only minor differences, demonstrating stable performance across different hardware.

**Environmental robustness:** AMT+ effectively mitigated interference effects of static objects, achieving ranging errors of less than 0.62 cm under various environmental complexities.

**Noise resilience:** AMT+ exhibited similar errors at quiet, medium, and high noise levels, all below 0.53 cm.

### B. 2-D Multi-Finger Tracking

**Single-target tracking:** Average tracking error of 0.54 cm with maximum error of 1.2 cm.

**Dual-target tracking:** Average tracking errors of 1.27 cm and 1.65 cm for intersecting and non-intersecting trajectories respectively.

**Triple-target tracking:** Average tracking error of 2.13 cm.

### C. Power Consumption and Response Delay

- **Power consumption:** 2.4% battery consumption per hour
- **Processing delay:** Less than 80 ms for 10*384-bit symbols

### D. Case Study

Three mobile applications were developed:

1. **Dual-player Pinball game:** Extended controlling area beyond phone sides
2. **Double-hand interactive album:** Supporting zoom, rotation, and switching gestures with 97.0% average accuracy
3. **Audio-Compatible air-gesture video switcher:** Video control through upward and downward hand swings

## X. Conclusion and Future Work

In this paper, an acoustic MIMO system called AMT+ has been proposed and implemented to achieve fine-grained multi-target tracking. AMT+ addresses the significant challenge of source separation through sliding-window correlation and deals with real-world targets as non-particles with multipath echoes. The system effectively resolves target confusion during multi-target tracking by exploiting distance projections between targets.

Extensive experiments demonstrated:
- Average error of 0.42 cm for 1-D absolute distance ranging
- Centimeter-level error for multi-target tracking
- 97.0% average classification accuracy for 14 controlling gestures

In future work, we will investigate multi-target localization in complex scenarios involving more reflections or interfering individuals, and apply the multi-target identification and localization algorithm to multi-target imaging.

## References

[1] J. Dai et al., "Effect of doppler shift on the performance of multicell full-duplex massive MIMO networks," IEEE Syst. J., vol. 14, no. 2, pp. 2421–2431, Jun. 2020.

[2] A. Zubow et al., "On phase offsets of 802.11ac commodity WiFi," arXiv:2005.03755, 2020.

[3] H. F. T. Ahmed et al., "Device free human gesture recognition using Wi-Fi CSI: A survey," Eng. Appl. Artif. Intell., vol. 87, 2020.

[4] Y. Liu et al., "Location, localization, and localizability," J. Comput. Sci. Technol., vol. 25, pp. 274–297, 2010.

[5] S. Yun et al., "Turning a mobile device into a mouse in the air," in Proc. 13th Annu. Int. Conf. Mobile Syst. Appl. Serv., 2015, pp. 15–29.

[... and many more references continue in the same format...]

---

**Manuscript Information:**
- Received: 15 November 2023
- Revised: 12 June 2024  
- Accepted: 18 June 2024
- Published: 21 June 2024
- Digital Object Identifier: 10.1109/TMC.2024.3417474

**Funding:** This research was funded by the National Natural Science Foundation of China and various other Chinese research foundations.

**Copyright:** © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.