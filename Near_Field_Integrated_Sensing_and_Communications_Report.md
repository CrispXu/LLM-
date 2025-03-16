
# Detailed Experiment Report: Near-Field Integrated Sensing and Communications

## 1. Introduction
The rapid evolution of wireless communication systems has led to an increasing demand for technologies that can handle multiple tasks simultaneously. One of the emerging paradigms is Near-Field Integrated Sensing and Communications (NF-ISC), which merges the capabilities of communication and sensing into a single system. This approach addresses the growing need for efficient data transmission and real-time environmental monitoring, such as location tracking, target detection, and environmental sensing.

Traditional wireless communication systems primarily focus on transmitting data, while sensing tasks (e.g., localization and environmental sensing) have often been handled separately by specialized systems. However, NF-ISC combines both tasks, enabling more efficient use of resources, improved accuracy, and faster response times. The near-field communication and sensing approach also provides better performance in dense environments, where users are located closer to the base station.

This report examines the theoretical framework, performance analysis, and optimization techniques for NF-ISC systems. Additionally, we will explore the simulation results that demonstrate the practical advantages of this unified system.

## 2. System Model and Framework

### 2.1 Near-Field vs. Far-Field Communication
The propagation characteristics of wireless communication signals are fundamentally determined by the distance and angle between the transmitter and receiver. In traditional far-field communication systems, the signal propagation typically follows the model of spherical waves, where the signal’s strength decreases with distance, and the angle between the transmitter and receiver significantly impacts the propagation. However, in near-field communication, both distance and angle play crucial roles in determining how the signal propagates.

#### Near-Field Propagation Characteristics:
In the near-field region, the distribution of electric and magnetic fields governs the signal propagation. According to the standard definition of near-field and far-field regions, when the distance between users and the base station is small (usually within a few wavelengths of the transmission frequency), the signal propagation is heavily influenced by the electric and magnetic fields. This characteristic enables the near-field communication to provide higher precision and resolution at short distances. Thus, location sensing and target detection perform exceptionally well in near-field environments. The near-field communication system allows distinguishing users that are located in the same direction but at different distances, a feat that far-field systems cannot easily achieve.

#### Far-Field Propagation Characteristics:
In contrast, far-field communication systems rely on the propagation of plane waves, where distance plays a secondary role, and angle becomes the primary factor influencing signal propagation. Signals in the far-field are often subject to attenuation, multipath interference, and signal refraction, leading to lower localization accuracy. Therefore, far-field communication systems typically face significant performance challenges in dense environments.

#### ![Figure 2.1-1 array response vector of Near-field and far-field](file:///5g/f1.png)
*Figure 2.1-1 array response vector of Near-field and far-field*

### 2.2 Joint Signal Transmission
One of the core innovations of NF-ISC is the joint transmission of communication and sensing signals. In traditional systems, communication and sensing are treated as separate processes, often requiring different resources. In NF-ISC, however, both tasks share the same time block, allowing for simultaneous communication and sensing operations.

The base station transmits a joint signal that carries both the communication data and the sensing information. This dual-purpose signal enables the base station to optimize the use of the available bandwidth and minimize interference between communication and sensing tasks.

The communication rate for user k can be represented by:

```
Rk = log2 (1 + Pk / (σ0 * d_k^α))
```

Where:
- `Rk` is the communication rate for user k,
- `Pk` is the transmission power,
- `σ0` is the noise power density,
- `d_k` is the distance from the base station to user k,
- `α` is the propagation loss factor.

To jointly carry out communication and sensing in the remaining time of the coherent time block, the BS transmits the following joint communication and sensing signal at time t:

```
X[t] = Σ fk * ck[t] + s[t]
```

Where:
- `fk` ∈ C^M×1 denotes the fully digital beamformer for conveying the information symbol `ck[t]` to the user k,
- `s[t]` denotes the dedicated sensing signal.

The covariance matrix:
```
Rx = E[X[t] * XH[t]] = Σ fk * fkH + Rs
```

### 3. Performance Analysis and Optimization

#### 3.1 Optimization Problem
The performance of the NF-ISC system heavily depends on the joint optimization of both communication and sensing tasks. The primary optimization goal is to minimize the Cramer-Rao Bound (CRB), which quantifies the uncertainty in parameter estimation (such as location sensing), while also ensuring that the communication rate remains above a specified threshold. The system must balance these two conflicting objectives, as improving one often results in a trade-off for the other.

Optimization Objectives:
- **Minimize CRB:** The CRB is an essential metric for evaluating the accuracy of parameter estimation, such as target localization or environmental sensing.
- **Maximize Communication Rate:** In communication tasks, the communication rate must meet a minimum requirement to ensure adequate data transmission.

The optimization problem can be expressed as:
```
min CRB(perception accuracy) + max Rk(communication rate)
```

#### 3.2 MUSIC Algorithm for Sensing
To improve sensing accuracy, NF-ISC uses the MUSIC (Multiple Signal Classification) algorithm for high-resolution parameter estimation. MUSIC is a subspace-based technique that exploits the eigenstructure of the signal covariance matrix to achieve super-resolution beyond conventional beamforming methods. It can be expressed as:
```
P_MUSIC(θ, φ) = 1 / ||A(θ, φ) * E(θ, φ)||
```

In NF-ISC, the MUSIC algorithm can be used for accurate localization of users or target detection, particularly in complex environments with multipath interference.

## 4. Simulation Results

### 4.1 Simulation Setup
The performance of the NF-ISC system was evaluated through simulations, which utilized the following key parameters:
- Number of Transmit Antennas (N): 65
- Transmit Power (P): 20 dBm
- Number of Users (K): 4
- Noise Power (σ): -60 dBm
- Minimum Communication Rate (Rmin): Specified for each user
- Target Distance (d): 20 m (near-field) and 80 m (far-field)
- Target Angle (θ): 45°
- Speed of Light (c): 3×10^8 m/s
- Carrier Frequency (f): 28 GHz
- Antenna Aperture (D): 0.5 m
- Antenna Spacing (d): d = D / (N-1)

### 4.2 Near-Field vs. Far-Field Performance
The simulations were conducted to compare the performance of NF-ISC in both near-field and far-field scenarios. The key observations were:
- **Near-Field Systems:** Near-field systems demonstrated significantly better localization accuracy and communication rates.
- **Far-Field Systems:** Far-field systems suffered from higher signal attenuation and multipath interference.

### 4.3 Simulation Insights
- **Improved Signal-to-Noise Ratio (SNR):** In near-field systems, the reduced propagation distance led to an improved SNR, resulting in better localization accuracy.
- **MUSIC Algorithm Effectiveness:** The MUSIC algorithm played a crucial role in improving the system’s sensing capabilities by providing high-resolution parameter estimation even in complex and noisy environments.

## 5. Results and Discussion

### 5.1 Communication and Sensing Performance
The simulation results highlight that near-field systems outperform far-field systems in both communication throughput and sensing accuracy.

### 5.2 Simulation Insights
Near-field systems benefit from the reduced propagation distance, which improves the signal-to-noise ratio (SNR) and localization accuracy.

## 6. Conclusion
The Near-Field Integrated Sensing and Communications (NF-ISC) framework represents a significant advancement in wireless communication technologies. Simulation results show that NF-ISC systems can achieve higher communication rates and sensing accuracy compared to traditional systems. Future research will focus on further optimizing joint communication and sensing strategies, as well as exploring practical implementation in real-world environments.
