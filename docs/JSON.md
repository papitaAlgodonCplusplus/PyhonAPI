# Swagger API Documentation - Sample Responses

This document provides sample responses from the Swagger API endpoints used in the Fertilizer Calculator integration.

## 🔗 Base URL
```
http://162.248.52.111:8082
```

## [INFO] Authentication
All API calls require Bearer token authentication:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## 📚 Catalog API

### GET `/Catalog`
**Purpose**: Get all available catalogs

**Sample Response**:
```json
{
  "result": [
    {
      "id": 1,
      "name": "Catalogo UCR"
    },
    {
      "id": 2,
      "name": "cliente test"
    }
  ]
}
```

**Response Fields**:
- `id`: Unique catalog identifier
- `name`: Human-readable catalog name

---

## [INFO] Fertilizer API

### GET `/Fertilizer?CatalogId={catalog_id}&IncludeInactives={boolean}`
**Purpose**: Get all fertilizers from a specific catalog

**Parameters**:
- `CatalogId`: Catalog ID (e.g., 1)
- `IncludeInactives`: Include inactive fertilizers (true/false)

**Sample Response**:
```json
{
  "result": {
    "fertilizers": [
      {
        "catalogId": 1,
        "name": "Acido Nítrico DAC",
        "manufacturer": "No Definido",
        "isLiquid": false,
        "active": true,
        "id": 1,
        "dateCreated": "2024-09-30T17:05:09.393",
        "dateUpdated": null,
        "createdBy": 1,
        "updatedBy": null
      },
      {
        "catalogId": 1,
        "name": "Acido Fosfórico",
        "manufacturer": "No Definido",
        "isLiquid": false,
        "active": true,
        "id": 2,
        "dateCreated": "2024-09-30T17:05:09.393",
        "dateUpdated": null,
        "createdBy": 1,
        "updatedBy": null
      }
    ]
  }
}
```

**Response Fields**:
- `id`: Unique fertilizer identifier
- `catalogId`: Parent catalog ID
- `name`: Fertilizer name (Spanish names)
- `manufacturer`: Manufacturer name
- `isLiquid`: Boolean - liquid (true) or solid (false)
- `active`: Boolean - whether fertilizer is active
- `dateCreated`: ISO timestamp of creation
- `dateUpdated`: ISO timestamp of last update (null if never updated)
- `createdBy`: User ID who created the record
- `updatedBy`: User ID who last updated (null if never updated)

**Available Fertilizers** (Sample from test):
- Acido Nítrico DAC
- Acido Fosfórico  
- Acido Sulfurico
- Nitrato de amonio
- Sulfato de amonio
- Nitrato de calcio
- *(and 26 more fertilizers)*

---

## 🧬 Fertilizer Chemistry API

### GET `/FertilizerChemistry?FertilizerId={fertilizer_id}&CatalogId={catalog_id}`
**Purpose**: Get detailed chemistry composition for a specific fertilizer

**Parameters**:
- `FertilizerId`: Fertilizer ID from `/Fertilizer` endpoint
- `CatalogId`: Catalog ID

**Expected Response Structure**:
```json
{
  "result": {
    "fertilizerChemistries": [
      {
        "fertilizerId": 1,
        "catalogId": 1,
        "formula": "HNO3",
        "purity": 65,
        "solubility": 1000,
        "density": 1.4,
        "isPhAdjuster": true,
        "ca": 0,
        "k": 0,
        "mg": 0,
        "na": 0,
        "nH4": 0,
        "nO3": 46.4,
        "sO4": 0,
        "cl": 0,
        "h2PO4": 0,
        "hcO3": 0,
        "molecularWeight": 63.01,
        "saltWeight": 62.00
      }
    ]
  }
}
```

**Chemistry Fields**:
- `formula`: Chemical formula (e.g., "HNO3", "KNO3")
- `purity`: Purity percentage (0-100)
- `solubility`: Solubility in water (g/L)
- `density`: Density (g/mL)
- `isPhAdjuster`: Boolean - affects pH
- **Cation concentrations (mg/g)**:
  - `ca`: Calcium
  - `k`: Potassium
  - `mg`: Magnesium
  - `na`: Sodium
  - `nH4`: Ammonium
- **Anion concentrations (mg/g)**:
  - `nO3`: Nitrate-Nitrogen
  - `sO4`: Sulfate-Sulfur
  - `cl`: Chloride
  - `h2PO4`: Phosphate-Phosphorus
  - `hcO3`: Bicarbonate
- `molecularWeight`: Molecular weight (g/mol)
- `saltWeight`: Salt weight (g/mol)

---

## [TARGET] Crop Phase Solution Requirements API

### GET `/CropPhaseSolutionRequirement/GetByPhaseId?PhaseId={phase_id}`
**Purpose**: Get target nutrient concentrations for a crop growth phase

**Parameters**:
- `PhaseId`: Crop phase ID (e.g., 1)

**Sample Response**:
```json
{
  "result": {
    "cropPhaseSolutionRequirement": {
      "phaseId": 1,
      "ec": 2.4,
      "hcO3": 30.5,
      "nO3": 750,
      "h2PO4": 193.97468,
      "sO4": 240.154,
      "cl": 0,
      "nH4": 0,
      "k": 390.98,
      "ca": 180.36,
      "mg": 24.305,
      "na": 0,
      "fe": 2.23388,
      "b": 0.3243,
      "cu": 0.06354,
      "zn": 0.32685,
      "mn": 0.27469,
      "mo": 0.09594,
      "n": 169.5,
      "s": 80.211436,
      "p": 62.0718976,
      "active": true,
      "id": 1,
      "dateCreated": "2024-09-13T11:49:27.81",
      "dateUpdated": null,
      "createdBy": 1,
      "updatedBy": null
    }
  }
}
```

**Requirement Fields**:
- `phaseId`: Crop phase identifier
- `ec`: Target electrical conductivity (dS/m)
- **Major nutrients (mg/L)**:
  - `ca`: Calcium target
  - `k`: Potassium target
  - `mg`: Magnesium target
  - `na`: Sodium target
  - `nH4`: Ammonium target
  - `nO3`: Nitrate-Nitrogen target
  - `sO4`: Sulfate-Sulfur target
  - `cl`: Chloride target
  - `h2PO4`: Phosphate-Phosphorus target
  - `hcO3`: Bicarbonate target
- **Micronutrients (mg/L)**:
  - `fe`: Iron
  - `b`: Boron
  - `cu`: Copper
  - `zn`: Zinc
  - `mn`: Manganese
  - `mo`: Molybdenum
- **Derived nutrients (mg/L)**:
  - `n`: Total Nitrogen
  - `s`: Total Sulfur
  - `p`: Total Phosphorus

---

## [WATER] Water Chemistry API

### GET `/WaterChemistry?WaterId={water_id}&CatalogId={catalog_id}`
**Purpose**: Get water source analysis data

**Parameters**:
- `WaterId`: Water source ID (e.g., 1)
- `CatalogId`: Catalog ID

**Sample Response**:
```json
{
  "result": {
    "waterChemistries": [
      {
        "waterId": 1,
        "ca": 10.15,
        "k": 2.6,
        "mg": 4.8,
        "na": 9.4,
        "nH4": 0,
        "fe": 0,
        "cu": 0.1,
        "mn": 0,
        "zn": 0.1,
        "nO3": 1.4,
        "sO4": 0,
        "cl": 1.2,
        "b": 0,
        "h2PO4": 0,
        "hcO3": 77,
        "bO4": 0,
        "moO4": 0.01,
        "ec": 0.15,
        "pH": 7.2,
        "analysisDate": "2022-12-08T00:00:00",
        "active": true,
        "id": 2,
        "dateCreated": "2024-09-13T10:59:19.757",
        "dateUpdated": null,
        "createdBy": 1,
        "updatedBy": null
      }
    ]
  }
}
```

**Water Analysis Fields**:
- `waterId`: Water source identifier
- **Major nutrients (mg/L) - current levels in water**:
  - `ca`: Calcium
  - `k`: Potassium
  - `mg`: Magnesium
  - `na`: Sodium
  - `nH4`: Ammonium
  - `nO3`: Nitrate-Nitrogen
  - `sO4`: Sulfate-Sulfur
  - `cl`: Chloride
  - `h2PO4`: Phosphate-Phosphorus
  - `hcO3`: Bicarbonate
- **Micronutrients (mg/L)**:
  - `fe`: Iron
  - `cu`: Copper
  - `mn`: Manganese
  - `zn`: Zinc
  - `b`: Boron
  - `bO4`: Borate
  - `moO4`: Molybdate
- **Water quality**:
  - `ec`: Electrical conductivity (dS/m)
  - `pH`: pH level
  - `analysisDate`: Date of water analysis (ISO format)

---

## [INFO] API Usage Workflow

### Typical Integration Flow:
1. **Get Catalogs**: `GET /Catalog` → Select catalog ID
2. **Get Fertilizers**: `GET /Fertilizer?CatalogId=1` → Get fertilizer list
3. **Get Chemistry**: `GET /FertilizerChemistry?FertilizerId=X&CatalogId=1` → For each fertilizer
4. **Get Requirements**: `GET /CropPhaseSolutionRequirement/GetByPhaseId?PhaseId=1` → Target values
5. **Get Water Analysis**: `GET /WaterChemistry?WaterId=1&CatalogId=1` → Current water levels
6. **Calculate**: Send all data to fertilizer calculator API

### Data Mapping for Calculator:
- **Fertilizers**: Combine `/Fertilizer` names with `/FertilizerChemistry` compositions
- **Targets**: Map requirement fields to calculator target concentrations
- **Water**: Map water chemistry to calculator water analysis
- **Calculate**: Optimize fertilizer dosages to meet targets

---

## [WARNING] Known Issues

### Missing Fertilizer Chemistry Data
The test results show fertilizers with zero nutrients because:
- `/Fertilizer` endpoint only provides metadata (name, manufacturer, etc.)
- **Need to call `/FertilizerChemistry` for each fertilizer** to get actual Ca, K, Mg, etc. values
- Current integration skips this step, resulting in zero dosages

### Field Name Mapping
- API uses lowercase field names: `ca`, `k`, `mg`, `nO3`, `sO4`, etc.
- Calculator expects mixed case: `Ca`, `K`, `Mg`, `NO3_N`, `SO4_S`, etc.
- Mapping is handled in the integration code

### Units
- All nutrient concentrations are in **mg/L**
- EC (electrical conductivity) is in **dS/m**
- pH is unitless (0-14 scale)
- Dates are in **ISO 8601 format**

---

## [INFO] Next Steps

1. **Implement FertilizerChemistry calls** for each fertilizer
2. **Test with complete nutrient data** 
3. **Validate calculation results** against expected fertilizer dosages
4. **Add error handling** for missing chemistry data
5. **Consider caching** chemistry data for performance