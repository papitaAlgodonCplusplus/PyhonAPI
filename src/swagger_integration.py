#!/usr/bin/env python3
"""
COMPLETE SWAGGER INTEGRATION MODULE
Real API calls, authentication, and data mapping
"""

import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from models import Fertilizer, FertilizerComposition, FertilizerChemistry
from fertilizer_database import FertilizerDatabase

class SwaggerAPIClient:
    """Complete Swagger API client with real authentication and data fetching"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.auth_token = None
        self.headers = {'Content-Type': 'application/json'}
        self.fertilizer_db = FertilizerDatabase()
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def login(self, user_email: str, password: str) -> Dict[str, Any]:
        """
        Real login implementation with proper authentication
        """
        print(f"🔐 Authenticating with {self.base_url}...")
        
        url = f"{self.base_url}/Authentication/Login"
        login_data = {
            "userEmail": user_email,
            "password": password
        }

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.post(url, json=login_data, headers=self.headers, timeout=30) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    try:
                        data = await response.json()
                        
                        if data.get('success') and data.get('result'):
                            self.auth_token = data['result']['token']
                            self.headers['Authorization'] = f'Bearer {self.auth_token}'
                            
                            print(f"✅ Authentication successful!")
                            print(f"📋 User: {data['result'].get('userName', 'Unknown')}")
                            print(f"🏢 Company: {data['result'].get('companyName', 'Unknown')}")
                            
                            return {
                                'success': True,
                                'token': self.auth_token,
                                'user_data': data['result']
                            }
                        else:
                            error_msg = data.get('message', 'Authentication failed')
                            print(f"❌ Login failed: {error_msg}")
                            raise Exception(f"Login failed: {error_msg}")
                            
                    except Exception as json_error:
                        print(f"❌ JSON parsing error: {json_error}")
                        print(f"Raw response: {response_text[:500]}")
                        raise Exception(f"Authentication response parsing failed: {json_error}")
                        
                else:
                    print(f"❌ HTTP Error {response.status}")
                    print(f"Response: {response_text[:500]}")
                    raise Exception(f"Authentication failed: HTTP {response.status}")
                    
        except aiohttp.ClientError as e:
            print(f"❌ Network error during authentication: {e}")
            raise Exception(f"Network error: {e}")
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            raise

    async def get_fertilizers(self, catalog_id: int, include_inactives: bool = False) -> List[Dict[str, Any]]:
        """
        Get all fertilizers from the specified catalog
        """
        if not self.auth_token:
            raise Exception("Authentication required - please login first")

        print(f"🌱 Fetching fertilizers from catalog {catalog_id}...")
        
        url = f"{self.base_url}/Fertilizer"
        params = {
            'CatalogId': catalog_id,
            'IncludeInactives': 'true' if include_inactives else 'false'
        }

        try:
            async with self.session.get(url, params=params, headers=self.headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    fertilizers = data.get('result', {}).get('fertilizers', [])
                    
                    print(f"✅ Found {len(fertilizers)} fertilizers")
                    
                    # Log first few fertilizer names for verification
                    for i, fert in enumerate(fertilizers[:5]):
                        print(f"  {i+1}. {fert.get('name', 'Unknown')} (ID: {fert.get('id', 'N/A')})")
                    
                    if len(fertilizers) > 5:
                        print(f"  ... and {len(fertilizers) - 5} more")
                    
                    return fertilizers
                    
                elif response.status == 401:
                    print(f"❌ Authentication failed - token may have expired")
                    raise Exception("Authentication failed - token may have expired")
                    
                else:
                    error_text = await response.text()
                    print(f"❌ HTTP {response.status}: {error_text[:300]}")
                    raise Exception(f"Failed to fetch fertilizers: HTTP {response.status}")
                    
        except aiohttp.ClientError as e:
            print(f"❌ Network error fetching fertilizers: {e}")
            raise Exception(f"Network error: {e}")

    async def get_fertilizer_chemistry(self, fertilizer_id: int, catalog_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed fertilizer chemistry data
        """
        if not self.auth_token:
            raise Exception("Authentication required")

        url = f"{self.base_url}/FertilizerChemistry"
        params = {
            'FertilizerId': fertilizer_id,
            'CatalogId': catalog_id
        }

        try:
            async with self.session.get(url, params=params, headers=self.headers, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    chemistry_list = data.get('result', {}).get('fertilizerChemistries', [])
                    
                    if chemistry_list:
                        chemistry = chemistry_list[0]
                        print(f"    ✅ Chemistry data: {chemistry.get('formula', 'N/A')}")
                        return chemistry
                    else:
                        print(f"    ⚠️  No chemistry data found")
                        return None
                        
                elif response.status == 404:
                    print(f"    ⚠️  Chemistry not found for fertilizer {fertilizer_id}")
                    return None
                    
                else:
                    print(f"    ⚠️  Chemistry fetch failed: HTTP {response.status}")
                    return None
                    
        except aiohttp.ClientError as e:
            print(f"    ⚠️  Network error fetching chemistry: {e}")
            return None
        except Exception as e:
            print(f"    ⚠️  Error fetching chemistry: {e}")
            return None

    async def get_crop_phase_requirements(self, phase_id: int) -> Optional[Dict[str, Any]]:
        """
        Get crop phase solution requirements
        """
        if not self.auth_token:
            raise Exception("Authentication required")

        print(f"🌾 Fetching crop phase requirements for phase {phase_id}...")
        
        url = f"{self.base_url}/CropPhaseSolutionRequirement/GetByPhaseId"
        params = {'PhaseId': phase_id}

        try:
            async with self.session.get(url, params=params, headers=self.headers, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    requirements = data.get('result', {}).get('cropPhaseSolutionRequirement')
                    
                    if requirements:
                        print(f"✅ Found crop requirements")
                        # Log some key requirements
                        for key in ['n', 'p', 'k', 'ca', 'mg']:
                            if key in requirements:
                                print(f"  {key.upper()}: {requirements[key]} mg/L")
                        return requirements
                    else:
                        print(f"⚠️  No requirements found for phase {phase_id}")
                        return None
                        
                else:
                    error_text = await response.text()
                    print(f"⚠️  Requirements fetch failed: HTTP {response.status}")
                    return None
                    
        except aiohttp.ClientError as e:
            print(f"❌ Network error fetching requirements: {e}")
            return None

    async def get_water_chemistry(self, water_id: int, catalog_id: int) -> Optional[Dict[str, Any]]:
        """
        Get water chemistry analysis
        """
        if not self.auth_token:
            raise Exception("Authentication required")

        print(f"💧 Fetching water chemistry for water {water_id}...")
        
        url = f"{self.base_url}/WaterChemistry"
        params = {
            'WaterId': water_id,
            'CatalogId': catalog_id
        }

        try:
            async with self.session.get(url, params=params, headers=self.headers, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    water_list = data.get('result', {}).get('waterChemistries', [])
                    
                    if water_list:
                        water_data = water_list[0]
                        print(f"✅ Found water analysis")
                        # Log some key parameters
                        for key in ['ca', 'k', 'mg', 'nO3', 'sO4']:
                            if key in water_data:
                                print(f"  {key}: {water_data[key]} mg/L")
                        return water_data
                    else:
                        print(f"⚠️  No water analysis found for water {water_id}")
                        return None
                        
                else:
                    error_text = await response.text()
                    print(f"⚠️  Water analysis fetch failed: HTTP {response.status}")
                    return None
                    
        except aiohttp.ClientError as e:
            print(f"❌ Network error fetching water analysis: {e}")
            return None

    def map_swagger_fertilizer_to_model(self, swagger_fert: Dict[str, Any], chemistry: Optional[Dict[str, Any]] = None) -> Fertilizer:
        """
        Convert Swagger fertilizer data to our Fertilizer model with intelligent composition mapping
        """
        name = swagger_fert.get('name', 'Unknown')
        print(f"    🔄 Mapping fertilizer: {name}")

        # Get chemistry data or use defaults
        if chemistry is None:
            chemistry = {
                'formula': name,
                'purity': 98.0,
                'density': 1.0,
                'solubility20': 100.0,
                'isPhAdjuster': False
            }

        # Extract basic properties
        formula = chemistry.get('formula', name)
        purity = float(chemistry.get('purity', 98.0))
        density = float(chemistry.get('density', 1.0))
        solubility = float(chemistry.get('solubility20', 100.0))
        is_ph_adjuster = bool(chemistry.get('isPhAdjuster', False))

        print(f"      Formula: {formula}")
        print(f"      Purity: {purity}%")
        print(f"      Density: {density}")

        # Get composition from our database using intelligent matching
        composition_data = self.fertilizer_db.find_fertilizer_composition(name, formula)

        if composition_data:
            cations = composition_data['cations'].copy()
            anions = composition_data['anions'].copy()
            molecular_weight = composition_data['mw']
            print(f"      ✅ Found in database: {composition_data['formula']}")
            
            # Calculate total content for verification
            total_content = sum(cations.values()) + sum(anions.values())
            print(f"      Total content: {total_content:.1f}%")
            
        else:
            # Create default empty composition with all required elements
            cations = {
                'Ca': 0.0, 'K': 0.0, 'Mg': 0.0, 'Na': 0.0, 'NH4': 0.0,
                'Fe': 0.0, 'Mn': 0.0, 'Zn': 0.0, 'Cu': 0.0
            }
            anions = {
                'N': 0.0, 'S': 0.0, 'Cl': 0.0, 'P': 0.0, 'HCO3': 0.0,
                'B': 0.0, 'Mo': 0.0
            }
            molecular_weight = 100.0
            print(f"      ⚠️  Not found in database, using defaults")

        # Validate and clean values
        if molecular_weight <= 0:
            molecular_weight = 100.0
        if purity <= 0 or purity > 100:
            purity = 98.0
        if density <= 0:
            density = 1.0
        if solubility < 0:
            solubility = 100.0

        # Create Fertilizer object
        fertilizer = Fertilizer(
            name=name,
            percentage=purity,
            molecular_weight=molecular_weight,
            salt_weight=molecular_weight,
            density=density,
            chemistry=FertilizerChemistry(
                formula=formula,
                purity=purity,
                solubility=solubility,
                is_ph_adjuster=is_ph_adjuster
            ),
            composition=FertilizerComposition(
                cations=cations,
                anions=anions
            )
        )

        # Log main nutrients for verification
        main_nutrients = []
        for elem, content in cations.items():
            if content > 1:
                main_nutrients.append(f"{elem}:{content:.1f}%")
        for elem, content in anions.items():
            if content > 1:
                main_nutrients.append(f"{elem}:{content:.1f}%")
        
        if main_nutrients:
            print(f"      Main nutrients: {', '.join(main_nutrients)}")
        else:
            print(f"      ⚠️  No significant nutrients found")

        return fertilizer

    def map_requirements_to_targets(self, requirements: Dict[str, Any]) -> Dict[str, float]:
        """
        Map Swagger crop requirements to our target concentrations format
        """
        print(f"🎯 Mapping crop requirements...")
        
        # Mapping from Swagger API field names to our element names
        element_mapping = {
            # Cations
            'ca': 'Ca',
            'k': 'K', 
            'mg': 'Mg',
            'na': 'Na',
            'nH4': 'NH4',
            'fe': 'Fe',
            'mn': 'Mn',
            'zn': 'Zn',
            'cu': 'Cu',
            
            # Anions  
            'n': 'N',
            'nO3': 'N',  # Nitrate as N
            's': 'S',
            'sO4': 'S',  # Sulfate as S
            'cl': 'Cl',
            'p': 'P',
            'h2PO4': 'P',  # Phosphate as P
            'hcO3': 'HCO3',
            'b': 'B',
            'mo': 'Mo'
        }

        targets = {}
        
        for api_field, our_field in element_mapping.items():
            if api_field in requirements:
                value = requirements[api_field]
                if value is not None and value > 0:
                    targets[our_field] = float(value)
                    print(f"  {our_field}: {value} mg/L")

        if not targets:
            print(f"  ⚠️  No valid requirements found, will use defaults")
        else:
            print(f"  ✅ Mapped {len(targets)} target concentrations")

        return targets

    def map_water_to_analysis(self, water: Dict[str, Any]) -> Dict[str, float]:
        """
        Map Swagger water data to our water analysis format
        """
        print(f"💧 Mapping water analysis...")
        
        # Mapping from Swagger API field names to our element names
        element_mapping = {
            # Cations
            'ca': 'Ca',
            'k': 'K',
            'mg': 'Mg', 
            'na': 'Na',
            'nH4': 'NH4',
            'fe': 'Fe',
            'mn': 'Mn',
            'zn': 'Zn',
            'cu': 'Cu',
            
            # Anions
            'nO3': 'N',     # Nitrate as N
            'sO4': 'S',     # Sulfate as S  
            'cl': 'Cl',
            'h2PO4': 'P',   # Phosphate as P
            'hcO3': 'HCO3',
            'b': 'B',
            'moO4': 'Mo'    # Molybdate as Mo
        }

        analysis = {}
        
        for api_field, our_field in element_mapping.items():
            if api_field in water:
                value = water[api_field]
                if value is not None and value >= 0:
                    analysis[our_field] = float(value)
                    print(f"  {our_field}: {value} mg/L")

        if not analysis:
            print(f"  ⚠️  No valid water analysis found, will use defaults")
        else:
            print(f"  ✅ Mapped {len(analysis)} water parameters")

        return analysis

    async def test_connection(self) -> bool:
        """
        Test connection to the Swagger API
        """
        print(f"🔍 Testing connection to {self.base_url}...")
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            # Try a simple endpoint that doesn't require authentication
            async with self.session.get(f"{self.base_url}/health", timeout=10) as response:
                if response.status == 200:
                    print(f"✅ API connection successful")
                    return True
                else:
                    print(f"⚠️  API responded with status {response.status}")
                    return False
                    
        except aiohttp.ClientError as e:
            print(f"❌ Connection failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Connection test error: {e}")
            return False

    def get_integration_summary(self, fertilizers_fetched: int, fertilizers_processed: int, 
                              targets_mapped: int, water_mapped: int) -> Dict[str, Any]:
        """
        Generate integration summary statistics
        """
        return {
            "integration_status": "complete",
            "api_endpoints_used": [
                "/Authentication/Login",
                "/Fertilizer",
                "/FertilizerChemistry", 
                "/CropPhaseSolutionRequirement/GetByPhaseId",
                "/WaterChemistry"
            ],
            "data_processing": {
                "fertilizers_fetched": fertilizers_fetched,
                "fertilizers_processed": fertilizers_processed,
                "processing_success_rate": f"{(fertilizers_processed/fertilizers_fetched*100):.1f}%" if fertilizers_fetched > 0 else "0%",
                "targets_mapped": targets_mapped,
                "water_parameters_mapped": water_mapped
            },
            "database_integration": {
                "fertilizer_database_used": True,
                "intelligent_pattern_matching": True,
                "fallback_compositions": True
            },
            "authentication": {
                "method": "Bearer Token",
                "status": "authenticated" if self.auth_token else "not_authenticated"
            }
        }

# Helper functions for standalone usage
async def test_swagger_integration():
    """
    Test function to verify Swagger integration works
    """
    print("🧪 Testing Swagger API Integration...")
    
    try:
        async with SwaggerAPIClient("http://162.248.52.111:8082") as client:
            # Test connection
            if not await client.test_connection():
                print("❌ Connection test failed")
                return False
            
            # Test authentication
            login_result = await client.login("csolano@iapcr.com", "123")
            if not login_result.get('success'):
                print("❌ Authentication test failed")
                return False
                
            # Test data fetching
            fertilizers = await client.get_fertilizers(1)
            if not fertilizers:
                print("❌ Fertilizer fetch test failed")
                return False
                
            print(f"✅ Integration test successful!")
            print(f"   - Fetched {len(fertilizers)} fertilizers")
            print(f"   - Authentication working")
            print(f"   - API connection stable")
            
            return True
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_swagger_integration())