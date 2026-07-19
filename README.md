# ZeroTrust Health

![Project Logo](public/ZeroTrustHealthLogo.png) 

ZeroTrust Health is a program that demonstrates the power of blind computation in a healthcare setting. Making use of Nillion's multiparty computation, the program provides a breast cancer image classification test performed over multiple health providers.

## Problem:
The article *"Sharing Is Caring-Data Sharing Initiative in Healthcare"*[[1]](#1) highlights a concern in regards to who gets access to all the health data. High valued data is expected to further grow in value due to advances in AI/ML and other emerging technologies. Do you know if your health provider can provide you the best care when you need it?

## Objective: 
Make use of modern cryptography to prevent barriers in the healthcare industry and ensure the world's most important data type, medical imaging, can be shared in a way that is fair and secure.

# Demo
![Diagram](public/Diagram.png)

The diagram illustrates the program's demo. There are two primary sections, a test computed over a full dataset and a test computed on Nillion's protocol on combining multiple datasets.

### Full Dataset Test
- 550+ image data instances
- 30 parameters and 1 target value (Diagnosis)
- 80/20 Split, (80) Training data, (20) Testing
- 3 Figures of Plot Distributions - Full, Small Subset, Large Subset
- Simple classification model use for computing thetas and test predictions 
- Single randomly selected test instance used, not a full test evaluation. Do not use program for predictions. Purpose of testing is to compare calculated values.

### Multi Party Nillion Test
- Small Subset (25%) and Large Subset (75%)
- 30 parameters and 1 target value (Diagnosis)
- Weighted average method used when combining thetas from subsets
- A scaling factor was applied to satisfy integer requirements in Nillion's Nada program

# About the program

## ✅ Secure Multi-Party Computation (Blind computation)

- [Nillion](https://nillion.com/) - Nillion is a secure computation network that decentralizes trust for high value data in the same way that blockchains decentralized transactions.

## ✅ Diagnostic Imaging Cancer Data

- [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) - The dataset used in the program is related to diagnostic imaging data focused on breast cancer classification. As included in the site's information, the features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. It describes characteristics of the cell nuclei present in the image.

## ✅ Files
- Main Python Program
- Nillion Nada Program
- Dataset

## ✅ Installation & Setup

### Requirements
- Python 3.11+
- Terminal

**1. Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/ZeroTrust-Health.git
cd ZeroTrust-Health
```

**2. Setup Configuration**
Copy the example environment variables file and fill in your Nillion credentials if you have them:
```bash
cp .env.example .env
```

**3. Setup Python Environment**
Create a virtual environment and install dependencies:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**4. Run the Machine Learning Application**
```bash
cd healthcare_imaging_compute
python3 healthcare_imaging_compute.py --disable_plot
```

*(Note: Nillion MPC cryptographic network execution requires the Nillion SDK `nilup`, which may be unavailable if Nillion servers are down.)*

## ✅ References
<a id="1">[1]</a>
Hulsen T. Sharing Is Caring-Data Sharing Initiatives in Healthcare. Int J Environ Res Public Health. 2020 Apr 27;17(9):3046. doi: 10.3390/ijerph17093046. PMID: 32349396; PMCID: PMC7246891.
