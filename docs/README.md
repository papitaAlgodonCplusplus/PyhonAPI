# [INFO] Fertilizer Calculator API

A Python FastAPI application that calculates optimal fertilizer dosages for hydroponic/fertigation systems. This API integrates with your Swagger API to retrieve fertilizer catalogs, crop requirements, and water analysis data.

## [TARGET] Features

- **Fertilizer Optimization**: Calculate optimal dosages for multiple fertilizers
- **Nutrient Balance**: Ensure target concentrations for all essential elements
- **Water Integration**: Account for existing water chemistry
- **Multiple Units**: Support for mg/L, mmol/L, and meq/L calculations
- **Swagger Integration**: Direct integration with your existing API
- **Real-time Calculations**: Fast optimization using scipy
- **Comprehensive Results**: Detailed breakdown of contributions and final solution

## 📋 Supported Elements

- **Cations**: Ca, K, Mg, Na, NH₄
- **Anions**: NO₃-N, SO₄-S, Cl⁻, H₂PO₄-P, HCO₃⁻
- **Additional**: EC, pH calculations

## [INFO] Quick Start

### 1. Setup

```bash
# Clone or download the files
git clone <your-repo-url>
cd fertilizer-calculator

# Make startup script executable
chmod +x startup.sh

# Run setup
./startup.sh
```

### 2. Start the API

```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
python fertilizer_calculator_api.py
```

The API will be available at `http://localhost:8000`

### 3. Test the Integration

```bash
# In a new terminal
python test_client.py
```

## 📖 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## [INFO] Configuration

### Swagger API Connection

Edit the `SwaggerAPIClient` configuration in `fertilizer_calculator_api.py`:

```python
api_client = SwaggerAPIClient(
    base_url="http://162.248.52.111:8082",
    auth_token="your-jwt-token-here"  # Add your JWT token
)
```

### Authentication

If your Swagger API requires authentication:

1. Get your JWT token from `/Authentication/Login`
2. Add it to the `auth_token` parameter
3. The client will automatically include it in headers

## [SECTION] Usage Examples

### Basic Calculation Request

```python
import requests

request_data = {
    "fertilizers": [
        {
            "name": "KNO3",
            "percentage": 98,
            "molecular_weight": 101.1,
            "chemistry": {
                "formula": "KNO3",
                "purity": 98,
                "solubility": 316,
                "is_ph_adjuster": False
            },
            "composition": {
                "cations": {"K": 38.7},
                "anions": {"NO3_N": 13.9}
            }
        }
    ],
    "target_concentrations": {
        "Ca": 172, "K": 260, "Mg": 50,
        "NO3_N": 166, "SO4_S": 41
    },
    "water_analysis": {
        "Ca": 10.2, "K": 2.6, "Mg": 4.8
    }
}

response = requests.post(
    "http://localhost:8000/calculate",
    json=request_data
)

result = response.json()
print(f"KNO3 dosage: {result['fertilizer_dosages']['KNO3']['dosage_g_per_L']} g/L")
```

### Using Swagger Integration

```python
# Test the Swagger API integration
response = requests.get("http://localhost:8000/test-swagger-integration")
print(response.json())
```

## 📈 Response Format

```json
{
  "fertilizer_dosages": {
    "KNO3": {
      "dosage_ml_per_L": 0.5,
      "dosage_g_per_L": 0.5
    }
  },
  "nutrient_contributions": {
    "APORTE_mg_L": {"Ca": 162, "K": 258},
    "DE_mmol_L": {"Ca": 4.04, "K": 6.59},
    "IONES_meq_L": {"Ca": 8.1, "K": 6.59}
  },
  "final_solution": {
    "FINAL_mg_L": {"Ca": 172.2, "K": 260.6},
    "calculated_EC": 1.94,
    "calculated_pH": 6.2
  }
}
```

## [INFO] Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/calculate` | POST | Calculate fertilizer dosages |
| `/test-swagger-integration` | GET | Test Swagger API connection |
| `/docs` | GET | API documentation |
| `/health` | GET | Health check |

## [INFO] Testing

### Run All Tests

```bash
python test_client.py
```

### Test Specific Components

```python
# Test only Swagger connection
tester = FertilizerAPITester()
await tester.test_swagger_connection()

# Test only calculator
await tester.run_fallback_test()
```

## 🔍 Troubleshooting

### Common Issues

1. **Swagger API Connection Failed**
   ```
   [FAILED] Failed to connect to Swagger API: Connection refused
   ```
   - Check if the Swagger API is running
   - Verify the URL: `http://162.248.52.111:8082`
   - Check authentication token

2. **Calculator API Not Running**
   ```
   [FAILED] Calculator API not running
   ```
   - Start the API: `python fertilizer_calculator_api.py`
   - Check port 8000 is available

3. **Optimization Failed**
   ```
   [FAILED] Calculation error: Optimization failed
   ```
   - Check fertilizer compositions are valid
   - Ensure target concentrations are achievable
   - Verify water analysis data

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 API Integration Examples

### Get Fertilizers from Swagger

```python
async def get_fertilizers():
    client = SwaggerAPIClient("http://162.248.52.111:8082", "your-token")
    fertilizers = await client.get_fertilizers(catalog_id=1)
    return fertilizers
```

### Get Target Concentrations

```python
async def get_targets():
    client = SwaggerAPIClient("http://162.248.52.111:8082", "your-token")
    requirements = await client.get_crop_phase_requirements(phase_id=1)
    return requirements
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## [INFO] License

MIT License - see LICENSE file for details

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Run the test client for diagnostics
4. Create an issue with error logs

---

## [TARGET] Expected Excel Output Mapping

| Excel Column | API Response Field | Units |
|-------------|-------------------|-------|
| APORTE mg/L | `nutrient_contributions.APORTE_mg_L` | mg/L |
| DE mmol/L | `nutrient_contributions.DE_mmol_L` | mmol/L |
| IONES meq/L | `nutrient_contributions.IONES_meq_L` | meq/L |
| DEL AGUA mg/L | `water_contribution.IONES_mg_L_DEL_AGUA` | mg/L |
| FINAL mg/L | `final_solution.FINAL_mg_L` | mg/L |
| EC | `final_solution.calculated_EC` | dS/m |
| pH | `final_solution.calculated_pH` | - |

This API replicates your Excel calculations with the same precision and methodology! [SUCCESS]