#!/usr/bin/env python3
"""
COMPLETE PDF GENERATOR MODULE
Professional PDF report generation with Excel-like tables
"""

from datetime import datetime
import os
from typing import Dict, List, Any, Optional

# PDF generation imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    print("WARNING: ReportLab not available. PDF generation disabled.")
    REPORTLAB_AVAILABLE = False


class ComprehensivePDFReportGenerator:
    """Professional PDF Report Generator for fertilizer calculations"""

    def __init__(self):
        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self.title_style = ParagraphStyle(
                'CustomTitle', parent=self.styles['Heading1'], fontSize=18, spaceAfter=30,
                alignment=1, textColor=colors.darkblue
            )
            self.subtitle_style = ParagraphStyle(
                'CustomSubtitle', parent=self.styles['Heading2'], fontSize=12, spaceAfter=20,
                alignment=1, textColor=colors.darkgreen
            )
        else:
            self.styles = None

    def generate_comprehensive_pdf(self, calculation_data: Dict[str, Any], filename: str = None) -> str:
        """Generate comprehensive PDF report with detailed Excel-like table"""

        if not REPORTLAB_AVAILABLE:
            print("WARNING: PDF generation skipped - ReportLab not available")
            # Create a simple text report instead
            return self._generate_text_report(calculation_data, filename)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/comprehensive_report_{timestamp}.pdf"

        # Ensure reports directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        print(f"Generating comprehensive PDF report: {filename}")

        doc = SimpleDocTemplate(
            filename,
            pagesize=landscape(A4),
            rightMargin=15, leftMargin=15,
            topMargin=25, bottomMargin=25
        )
        story = []

        title = Paragraph(
            "REPORTE DE CÁLCULO DE SOLUCIÓN NUTRITIVA", self.title_style)
        story.append(title)

        subtitle = Paragraph(
            "Sistema Avanzado de Optimización de Fertilizantes", self.subtitle_style)
        story.append(subtitle)
        story.append(Spacer(1, 20))

        # User information section (NEW SECTION)
        user_data = calculation_data.get('user_info', {})
        if user_data:
            print("Adding user information section to PDF...")
            user_info = self._create_user_info_section(user_data)
            story.extend(user_info)
            story.append(Spacer(1, 15))
        else:
            print("No user info provided, skipping user section")

        # Metadata section
        metadata = self._create_metadata_section(calculation_data)
        story.extend(metadata)
        story.append(Spacer(1, 25))

        # Main calculation table (Excel-like format)
        main_table = self._create_main_calculation_table(calculation_data)
        story.append(main_table)
        story.append(PageBreak())

        # Summary and analysis tables
        summary_tables = self._create_summary_tables(calculation_data)
        story.extend(summary_tables)

        # Build PDF
        try:
            doc.build(story)
            print(f"PDF report generated successfully: {filename}")
        except Exception as e:
            print(f"PDF generation failed: {e}")
            # Fallback to text report
            return self._generate_text_report(calculation_data, filename.replace('.pdf', '.txt'))

        return filename

    def _create_user_info_section(self, user_data: Dict[str, Any]) -> List:
        """Create user information section for PDF header"""
        if not REPORTLAB_AVAILABLE:
            return []

        elements = []
        
        # User info header
        user_title = Paragraph("INFORMACIÓN DEL USUARIO", self.subtitle_style)
        elements.append(user_title)
        elements.append(Spacer(1, 10))
        
        # Safely extract user data with defaults
        user_id = user_data.get('id', 'N/A')
        user_email = user_data.get('userEmail', 'N/A')
        client_id = user_data.get('clientId', 'N/A')
        profile_id = user_data.get('profileId', 'N/A')
        status_id = user_data.get('userStatusId', 'N/A')
        
        # Format creation date if available
        date_created = user_data.get('dateCreated', '')
        if date_created and len(date_created) >= 10:
            formatted_date = date_created[:10]  # Extract YYYY-MM-DD part
        else:
            formatted_date = 'N/A'
        
        # User data table
        user_table_data = [
            ['ID de Usuario:', str(user_id), 'Email:', str(user_email)],
            ['Cliente ID:', str(client_id), 'Perfil ID:', str(profile_id)],
            ['Estado ID:', str(status_id), 'Fecha Creación:', formatted_date]
        ]

        user_table = Table(user_table_data, colWidths=[2*inch, 2*inch, 2*inch, 2*inch])
        user_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),  # Bold first column
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),  # Bold third column
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.lightblue, colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(user_table)
        elements.append(Spacer(1, 20))
        return elements
    
    def _generate_text_report(self, calculation_data: Dict[str, Any], filename: str = None) -> str:
        """Generate a text-based report when PDF generation is not available"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/text_report_{timestamp}.txt"

        # Ensure it's a .txt file
        if not filename.endswith('.txt'):
            filename = filename.replace('.pdf', '.txt')

        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        print(f"📄 Generating text report: {filename}")

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("FERTILIZER CALCULATION REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Add user information section (NEW)
                user_data = calculation_data.get('user_info', {})
                if user_data:
                    f.write("USER INFORMATION:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"User ID: {user_data.get('id', 'N/A')}\n")
                    f.write(f"Email: {user_data.get('userEmail', 'N/A')}\n")
                    f.write(f"Client ID: {user_data.get('clientId', 'N/A')}\n")
                    f.write(f"Profile ID: {user_data.get('profileId', 'N/A')}\n")
                    f.write(f"Status ID: {user_data.get('userStatusId', 'N/A')}\n")
                    date_created = user_data.get('dateCreated', '')
                    if date_created:
                        f.write(f"Created: {date_created[:10] if len(date_created) >= 10 else date_created}\n")
                    f.write("\n")

                # Integration metadata
                metadata = calculation_data.get('integration_metadata', {})
                f.write("INTEGRATION METADATA:\n")
                f.write("-" * 25 + "\n")
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

                # Calculation results
                calc_results = calculation_data.get('calculation_results', {})

                # Fertilizer dosages
                fertilizer_dosages = calc_results.get('fertilizer_dosages', {})
                f.write("FERTILIZER DOSAGES:\n")
                f.write("-" * 20 + "\n")

                total_dosage = 0.0
                active_fertilizers = 0

                for name, dosage in fertilizer_dosages.items():
                    dosage_g_l = dosage.get(
                        'dosage_g_per_L', 0) if isinstance(dosage, dict) else 0
                    dosage_ml_l = dosage.get(
                        'dosage_ml_per_L', 0) if isinstance(dosage, dict) else 0

                    if dosage_g_l > 0:
                        f.write(
                            f"{name}: {dosage_g_l:.3f} g/L ({dosage_ml_l:.3f} mL/L)\n")
                        total_dosage += dosage_g_l
                        active_fertilizers += 1
                    else:
                        f.write(f"{name}: 0.000 g/L (not used)\n")

                f.write(f"\nSUMMARY:\n")
                f.write(f"Total dosage: {total_dosage:.3f} g/L\n")
                f.write(f"Active fertilizers: {active_fertilizers}\n")

                # Final solution
                final_solution = calc_results.get('final_solution', {})
                if final_solution:
                    f.write(f"\nFINAL SOLUTION:\n")
                    f.write("-" * 15 + "\n")

                    final_mg_l = final_solution.get('FINAL_mg_L', {})
                    for element, concentration in final_mg_l.items():
                        if concentration > 0.1:
                            f.write(f"{element}: {concentration:.1f} mg/L\n")

                    ec = final_solution.get('calculated_EC', 0)
                    ph = final_solution.get('calculated_pH', 0)
                    f.write(f"\nCalculated EC: {ec:.2f} dS/m\n")
                    f.write(f"Calculated pH: {ph:.1f}\n")

                # Ionic balance
                ionic_balance = calc_results.get('ionic_balance', {})
                if ionic_balance:
                    f.write(f"\nIONIC BALANCE:\n")
                    f.write("-" * 15 + "\n")
                    f.write(
                        f"Cation sum: {ionic_balance.get('cation_sum', 0):.2f} meq/L\n")
                    f.write(
                        f"Anion sum: {ionic_balance.get('anion_sum', 0):.2f} meq/L\n")
                    f.write(
                        f"Balance error: {ionic_balance.get('difference_percentage', 0):.1f}%\n")

                    is_balanced = ionic_balance.get('is_balanced', 0)
                    f.write(
                        f"Status: {'BALANCED' if is_balanced else 'UNBALANCED'}\n")

                # Cost analysis
                cost_analysis = calc_results.get('cost_analysis', {})
                if cost_analysis:
                    f.write(f"\nCOST ANALYSIS:\n")
                    f.write("-" * 15 + "\n")

                    total_cost = cost_analysis.get('total_cost_diluted', 0)
                    f.write(f"Total cost: ${total_cost:.2f}\n")

                    cost_per_fert = cost_analysis.get(
                        'cost_per_fertilizer', {})
                    if cost_per_fert:
                        f.write("Cost breakdown:\n")
                        for fert, cost in cost_per_fert.items():
                            if cost > 0:
                                f.write(f"  {fert}: ${cost:.3f}\n")

                f.write(f"\n" + "=" * 50 + "\n")
                f.write("Report generation completed successfully\n")

            print(f"✅ Text report generated successfully: {filename}")
            return filename

        except Exception as e:
            print(f"❌ Text report generation failed: {e}")
            # Create minimal fallback
            fallback_filename = f"reports/minimal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(fallback_filename, 'w') as f:
                f.write(
                    f"Fertilizer calculation completed at {datetime.now()}\n")
                f.write("Report generation encountered errors.\n")
            return fallback_filename

    def _create_metadata_section(self, calculation_data: Dict[str, Any]) -> List:
        """Create metadata section with calculation information"""
        if not REPORTLAB_AVAILABLE:
            return []

        elements = []
        metadata = calculation_data.get('integration_metadata', {})
        calc_results = calculation_data.get('calculation_results', {})
        final_solution = calc_results.get('final_solution', {})

        metadata_table_data = [
            ['Fecha y Hora:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             'Fuente de Datos:', metadata.get('data_source', 'API Integration')],
            ['Fertilizantes Analizados:', str(metadata.get('fertilizers_analyzed', 'N/A')),
             'Volumen de Solución:', '1000 L'],
            ['Método de Optimización:', metadata.get('method_used', 'Deterministic'),
             'Tipo de Cálculo:', 'Optimización Avanzada'],
            ['EC Final:', f"{final_solution.get('calculated_EC', 0):.2f} dS/m",
             'pH Final:', f"{final_solution.get('calculated_pH', 0):.1f}"]
        ]

        metadata_table = Table(metadata_table_data, colWidths=[
                               2*inch, 2*inch, 2*inch, 2*inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1),
             [colors.lightgrey, colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))

        elements.append(metadata_table)
        return elements

    def _create_main_calculation_table(self, calculation_data: Dict[str, Any]) -> object:
        """Create the main Excel-like calculation table with all fertilizer rows"""
        if not REPORTLAB_AVAILABLE:
            return None

        calc_results = calculation_data.get('calculation_results', {})
        fertilizer_dosages = calc_results.get('fertilizer_dosages', {})
        nutrient_contributions = calc_results.get('nutrient_contributions', {})
        water_contribution = calc_results.get('water_contribution', {})
        final_solution = calc_results.get('final_solution', {})

        # Define column headers exactly as specified
        headers = [
            'FERTILIZANTE', '% P', 'Peso molecular\n(Sal)', 'Peso molecular\n(Elem1)',
            'Peso molecular\n(Elem2)', 'Peso de sal\n(mg o ml/L)', 'Peso de sal\n(mmol/L)',
            'Ca', 'K', 'Mg', 'Na', 'NH4', 'NO3-', 'N', 'SO4=', 'S', 'Cl-',
            'H2PO4-', 'P', 'HCO3-', 'Σ aniones', 'CE'
        ]

        table_data = [headers]

        # Add fertilizer rows for active fertilizers only
        from fertilizer_database import FertilizerDatabase
        fertilizer_db = FertilizerDatabase()

        fertilizer_rows_added = 0
        for fert_name, dosage_info in fertilizer_dosages.items():
            if isinstance(dosage_info, dict):
                dosage_g_l = dosage_info.get('dosage_g_per_L', 0)
            elif hasattr(dosage_info, 'dosage_g_per_L'):
                dosage_g_l = dosage_info.dosage_g_per_L
            else:
                dosage_g_l = 0
            if dosage_g_l > 0:
                row = self._create_fertilizer_row(
                    fert_name, dosage_info, fertilizer_db)
                table_data.append(row)
                fertilizer_rows_added += 1
                print(
                    f"    Added fertilizer row: {fert_name} ({dosage_g_l:.3f} g/L)")

        print(f"    Total fertilizer rows added: {fertilizer_rows_added}")

        # Add summary rows
        summary_rows = self._create_comprehensive_summary_rows(
            nutrient_contributions, water_contribution, final_solution
        )
        table_data.extend(summary_rows)
        print(f"    Added {len(summary_rows)} summary rows")

        # Calculate number of fertilizer rows for styling
        num_fertilizer_rows = fertilizer_rows_added

        print(
            f"    Creating table with {len(table_data)} total rows ({num_fertilizer_rows} fertilizer + {len(summary_rows)} summary)")

        # Create table with professional styling
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),

            # Fertilizer rows styling (if any exist)
            ('FONTNAME', (0, 1), (-1, num_fertilizer_rows), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, num_fertilizer_rows), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, num_fertilizer_rows),
            [colors.white, colors.lightgrey]),

            # Summary rows styling
            ('BACKGROUND', (0, num_fertilizer_rows+1), (-1, -1), colors.lightyellow),
            ('FONTNAME', (0, num_fertilizer_rows+1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, num_fertilizer_rows+1), (-1, -1), 6),

            # Borders and alignment
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ]))

        return table

    def _create_fertilizer_row(self, fert_name: str, dosage_info, fertilizer_db) -> List:
        """Create a detailed table row for a single fertilizer with CORRECT calculations"""
        if isinstance(dosage_info, dict):
            dosage_g_l = dosage_info.get('dosage_g_per_L', 0)
        elif hasattr(dosage_info, 'dosage_g_per_L'):
            dosage_g_l = dosage_info.dosage_g_per_L
        else:
            dosage_g_l = 0

        print(f"      Creating row for {fert_name}: {dosage_g_l:.3f} g/L")

        # Get fertilizer composition from database
        composition_data = fertilizer_db.find_fertilizer_composition(
            fert_name, fert_name)

        if composition_data:
            molecular_weight = composition_data['mw']
            cations = composition_data['cations']
            anions = composition_data['anions']
            print(f"        Found composition: {composition_data['formula']}")
        else:
            molecular_weight = 100
            cations = {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0}
            anions = {'N': 0, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0}
            print(f"        Using default composition")

        dosage_mg_l = dosage_g_l * 1000
        dosage_mmol_l = dosage_mg_l / molecular_weight if molecular_weight > 0 else 0

        # Get main elements for molecular weight display
        main_elements = []
        all_nutrients = {**cations, **anions}
        sorted_nutrients = sorted(all_nutrients.items(),
                                key=lambda x: x[1], reverse=True)

        # Get atomic weights for main elements
        atomic_weights = {
            'Ca': 40.08, 'K': 39.10, 'Mg': 24.31, 'Na': 22.99, 'NH4': 18.04,
            'N': 14.01, 'S': 32.06, 'P': 30.97, 'Cl': 35.45, 'Fe': 55.85,
            'Mn': 54.94, 'Zn': 65.38, 'Cu': 63.55, 'B': 10.81, 'Mo': 95.96
        }

        for elem, content in sorted_nutrients:
            if content > 1 and elem in atomic_weights:
                main_elements.append((elem, atomic_weights[elem]))
                if len(main_elements) >= 2:
                    break

        elem1_weight = main_elements[0][1] if len(main_elements) > 0 else 0
        elem2_weight = main_elements[1][1] if len(main_elements) > 1 else 0

        # Calculate actual nutrient contributions (mg/L)
        purity_factor = 98.0 / 100.0  # Assume 98% purity

        nutrient_contributions = {}
        for elem in ['Ca', 'K', 'Mg', 'Na', 'NH4', 'N', 'S', 'Cl', 'P', 'HCO3']:
            cation_content = cations.get(elem, 0)
            anion_content = anions.get(elem, 0)
            total_content = cation_content + anion_content

            if total_content > 0:
                contribution = dosage_mg_l * \
                    (total_content / 100.0) * purity_factor
                nutrient_contributions[elem] = contribution
            else:
                nutrient_contributions[elem] = 0

        # Calculate anion sum and EC contribution
        anion_elements = ['N', 'S', 'Cl', 'P', 'HCO3']
        anion_sum = sum(nutrient_contributions.get(elem, 0)
                        for elem in anion_elements)

        # EC contribution (simplified calculation)
        ec_contribution = dosage_mmol_l * 0.1

        print(
            f"        Main contributions: Ca={nutrient_contributions.get('Ca', 0):.1f}, K={nutrient_contributions.get('K', 0):.1f}, N={nutrient_contributions.get('N', 0):.1f} mg/L")

        row = [
            fert_name,                                      # FERTILIZANTE
            "98.0",                                         # % P (purity)
            f"{molecular_weight:.1f}",                      # Peso molecular (Sal)
            # Peso molecular (Elem1)
            f"{elem1_weight:.1f}",
            # Peso molecular (Elem2)
            f"{elem2_weight:.1f}",
            # Peso de sal (mg o ml/L) - in g/L
            f"{dosage_g_l:.3f}",
            f"{dosage_mmol_l:.3f}",                        # Peso de sal (mmol/L)
            f"{nutrient_contributions.get('Ca', 0):.1f}",   # Ca contribution
            f"{nutrient_contributions.get('K', 0):.1f}",    # K contribution
            f"{nutrient_contributions.get('Mg', 0):.1f}",   # Mg contribution
            f"{nutrient_contributions.get('Na', 0):.1f}",   # Na contribution
            f"{nutrient_contributions.get('NH4', 0):.1f}",  # NH4 contribution
            f"{nutrient_contributions.get('N', 0):.1f}",    # NO3- contribution
            f"{nutrient_contributions.get('N', 0):.1f}",    # N contribution
            f"{nutrient_contributions.get('S', 0):.1f}",    # SO4= contribution
            f"{nutrient_contributions.get('S', 0):.1f}",    # S contribution
            f"{nutrient_contributions.get('Cl', 0):.1f}",   # Cl- contribution
            f"{nutrient_contributions.get('P', 0):.1f}",    # H2PO4- contribution
            f"{nutrient_contributions.get('P', 0):.1f}",    # P contribution
            f"{nutrient_contributions.get('HCO3', 0):.1f}",  # HCO3- contribution
            f"{anion_sum:.1f}",                            # Σ aniones
            f"{ec_contribution:.2f}"                       # CE contribution
        ]

        return row

    def _create_comprehensive_summary_rows(self, nutrient_contributions: Dict, water_contribution: Dict, final_solution: Dict) -> List[List]:
        """Create comprehensive summary rows matching Excel format"""
        summary_rows = []

        # Get data dictionaries
        aporte_mg = nutrient_contributions.get('APORTE_mg_L', {})
        aporte_mmol = nutrient_contributions.get('DE_mmol_L', {})
        aporte_meq = nutrient_contributions.get('IONES_meq_L', {})

        agua_mg = water_contribution.get('IONES_mg_L_DEL_AGUA', {})
        agua_mmol = water_contribution.get('mmol_L', {})
        agua_meq = water_contribution.get('meq_L', {})

        final_mg = final_solution.get('FINAL_mg_L', {})
        final_mmol = final_solution.get('FINAL_mmol_L', {})
        final_meq = final_solution.get('FINAL_meq_L', {})

        anion_elements = ['N', 'S', 'Cl', 'P', 'HCO3']
        final_ec = final_solution.get('calculated_EC', 0)

        # Row 1: Aporte de Iones (mg/L)
        aporte_anion_sum = sum(aporte_mg.get(elem, 0)
                               for elem in anion_elements)
        row1 = ['Aporte de Iones (mg/L)', '', '', '', '', '', '',
                f"{aporte_mg.get('Ca', 0):.1f}", f"{aporte_mg.get('K', 0):.1f}",
                f"{aporte_mg.get('Mg', 0):.1f}", f"{aporte_mg.get('Na', 0):.1f}",
                f"{aporte_mg.get('NH4', 0):.1f}", f"{aporte_mg.get('N', 0):.1f}",
                f"{aporte_mg.get('N', 0):.1f}", f"{aporte_mg.get('S', 0):.1f}",
                f"{aporte_mg.get('S', 0):.1f}", f"{aporte_mg.get('Cl', 0):.1f}",
                f"{aporte_mg.get('P', 0):.1f}", f"{aporte_mg.get('P', 0):.1f}",
                f"{aporte_mg.get('HCO3', 0):.1f}", f"{aporte_anion_sum:.1f}", f"{final_ec:.2f}"]

        # Row 2: Aporte de Iones (mmol/L)
        aporte_mmol_anion_sum = sum(aporte_mmol.get(elem, 0)
                                    for elem in anion_elements)
        row2 = ['Aporte de Iones (mmol/L)', '', '', '', '', '', '',
                f"{aporte_mmol.get('Ca', 0):.3f}", f"{aporte_mmol.get('K', 0):.3f}",
                f"{aporte_mmol.get('Mg', 0):.3f}", f"{aporte_mmol.get('Na', 0):.3f}",
                f"{aporte_mmol.get('NH4', 0):.3f}", f"{aporte_mmol.get('N', 0):.3f}",
                f"{aporte_mmol.get('N', 0):.3f}", f"{aporte_mmol.get('S', 0):.3f}",
                f"{aporte_mmol.get('S', 0):.3f}", f"{aporte_mmol.get('Cl', 0):.3f}",
                f"{aporte_mmol.get('P', 0):.3f}", f"{aporte_mmol.get('P', 0):.3f}",
                f"{aporte_mmol.get('HCO3', 0):.3f}", f"{aporte_mmol_anion_sum:.3f}", '']

        # Row 3: Aporte de Iones (meq/L)
        aporte_meq_anion_sum = sum(aporte_meq.get(elem, 0)
                                   for elem in anion_elements)
        row3 = ['Aporte de Iones (meq/L)', '', '', '', '', '', '',
                f"{aporte_meq.get('Ca', 0):.3f}", f"{aporte_meq.get('K', 0):.3f}",
                f"{aporte_meq.get('Mg', 0):.3f}", f"{aporte_meq.get('Na', 0):.3f}",
                f"{aporte_meq.get('NH4', 0):.3f}", f"{aporte_meq.get('N', 0):.3f}",
                f"{aporte_meq.get('N', 0):.3f}", f"{aporte_meq.get('S', 0):.3f}",
                f"{aporte_meq.get('S', 0):.3f}", f"{aporte_meq.get('Cl', 0):.3f}",
                f"{aporte_meq.get('P', 0):.3f}", f"{aporte_meq.get('P', 0):.3f}",
                f"{aporte_meq.get('HCO3', 0):.3f}", f"{aporte_meq_anion_sum:.3f}", '']

        # Row 4: Iones en Agua (mg/L)
        agua_anion_sum = sum(agua_mg.get(elem, 0) for elem in anion_elements)
        row4 = ['Iones en Agua (mg/L)', '', '', '', '', '', '',
                f"{agua_mg.get('Ca', 0):.1f}", f"{agua_mg.get('K', 0):.1f}",
                f"{agua_mg.get('Mg', 0):.1f}", f"{agua_mg.get('Na', 0):.1f}",
                f"{agua_mg.get('NH4', 0):.1f}", f"{agua_mg.get('N', 0):.1f}",
                f"{agua_mg.get('N', 0):.1f}", f"{agua_mg.get('S', 0):.1f}",
                f"{agua_mg.get('S', 0):.1f}", f"{agua_mg.get('Cl', 0):.1f}",
                f"{agua_mg.get('P', 0):.1f}", f"{agua_mg.get('P', 0):.1f}",
                f"{agua_mg.get('HCO3', 0):.1f}", f"{agua_anion_sum:.1f}", '']

        # Row 5: Iones en Agua (mmol/L)
        agua_mmol_anion_sum = sum(agua_mmol.get(elem, 0)
                                  for elem in anion_elements)
        row5 = ['Iones en Agua (mmol/L)', '', '', '', '', '', '',
                f"{agua_mmol.get('Ca', 0):.3f}", f"{agua_mmol.get('K', 0):.3f}",
                f"{agua_mmol.get('Mg', 0):.3f}", f"{agua_mmol.get('Na', 0):.3f}",
                f"{agua_mmol.get('NH4', 0):.3f}", f"{agua_mmol.get('N', 0):.3f}",
                f"{agua_mmol.get('N', 0):.3f}", f"{agua_mmol.get('S', 0):.3f}",
                f"{agua_mmol.get('S', 0):.3f}", f"{agua_mmol.get('Cl', 0):.3f}",
                f"{agua_mmol.get('P', 0):.3f}", f"{agua_mmol.get('P', 0):.3f}",
                f"{agua_mmol.get('HCO3', 0):.3f}", f"{agua_mmol_anion_sum:.3f}", '']

        # Row 6: Iones en Agua (meq/L)
        agua_meq_anion_sum = sum(agua_meq.get(elem, 0)
                                 for elem in anion_elements)
        row6 = ['Iones en Agua (meq/L)', '', '', '', '', '', '',
                f"{agua_meq.get('Ca', 0):.3f}", f"{agua_meq.get('K', 0):.3f}",
                f"{agua_meq.get('Mg', 0):.3f}", f"{agua_meq.get('Na', 0):.3f}",
                f"{agua_meq.get('NH4', 0):.3f}", f"{agua_meq.get('N', 0):.3f}",
                f"{agua_meq.get('N', 0):.3f}", f"{agua_meq.get('S', 0):.3f}",
                f"{agua_meq.get('S', 0):.3f}", f"{agua_meq.get('Cl', 0):.3f}",
                f"{agua_meq.get('P', 0):.3f}", f"{agua_meq.get('P', 0):.3f}",
                f"{agua_meq.get('HCO3', 0):.3f}", f"{agua_meq_anion_sum:.3f}", '']

        # Row 7: Iones en SONU Final (mg/L)
        final_anion_sum = sum(final_mg.get(elem, 0) for elem in anion_elements)
        row7 = ['Iones en SONU Final (mg/L)', '', '', '', '', '', '',
                f"{final_mg.get('Ca', 0):.1f}", f"{final_mg.get('K', 0):.1f}",
                f"{final_mg.get('Mg', 0):.1f}", f"{final_mg.get('Na', 0):.1f}",
                f"{final_mg.get('NH4', 0):.1f}", f"{final_mg.get('N', 0):.1f}",
                f"{final_mg.get('N', 0):.1f}", f"{final_mg.get('S', 0):.1f}",
                f"{final_mg.get('S', 0):.1f}", f"{final_mg.get('Cl', 0):.1f}",
                f"{final_mg.get('P', 0):.1f}", f"{final_mg.get('P', 0):.1f}",
                f"{final_mg.get('HCO3', 0):.1f}", f"{final_anion_sum:.1f}", f"{final_ec:.2f}"]

        # Row 8: Iones en SONU (mmol/L)
        final_mmol_anion_sum = sum(final_mmol.get(elem, 0)
                                   for elem in anion_elements)
        row8 = ['Iones en SONU (mmol/L)', '', '', '', '', '', '',
                f"{final_mmol.get('Ca', 0):.3f}", f"{final_mmol.get('K', 0):.3f}",
                f"{final_mmol.get('Mg', 0):.3f}", f"{final_mmol.get('Na', 0):.3f}",
                f"{final_mmol.get('NH4', 0):.3f}", f"{final_mmol.get('N', 0):.3f}",
                f"{final_mmol.get('N', 0):.3f}", f"{final_mmol.get('S', 0):.3f}",
                f"{final_mmol.get('S', 0):.3f}", f"{final_mmol.get('Cl', 0):.3f}",
                f"{final_mmol.get('P', 0):.3f}", f"{final_mmol.get('P', 0):.3f}",
                f"{final_mmol.get('HCO3', 0):.3f}", f"{final_mmol_anion_sum:.3f}", '']

        # Row 9: Iones en SONU (meq/L)
        final_meq_anion_sum = sum(final_meq.get(elem, 0)
                                  for elem in anion_elements)
        row9 = ['Iones en SONU (meq/L)', '', '', '', '', '', '',
                f"{final_meq.get('Ca', 0):.3f}", f"{final_meq.get('K', 0):.3f}",
                f"{final_meq.get('Mg', 0):.3f}", f"{final_meq.get('Na', 0):.3f}",
                f"{final_meq.get('NH4', 0):.3f}", f"{final_meq.get('N', 0):.3f}",
                f"{final_meq.get('N', 0):.3f}", f"{final_meq.get('S', 0):.3f}",
                f"{final_meq.get('S', 0):.3f}", f"{final_meq.get('Cl', 0):.3f}",
                f"{final_meq.get('P', 0):.3f}", f"{final_meq.get('P', 0):.3f}",
                f"{final_meq.get('HCO3', 0):.3f}", f"{final_meq_anion_sum:.3f}", '']

        summary_rows.extend(
            [row1, row2, row3, row4, row5, row6, row7, row8, row9])
        return summary_rows

    def _create_summary_tables(self, calculation_data: Dict[str, Any]) -> List:
        """Create additional summary and analysis tables"""
        if not REPORTLAB_AVAILABLE:
            return []

        elements = []
        calc_results = calculation_data.get('calculation_results', {})

        # Verification Results Table
        verification_results = calc_results.get('verification_results', [])
        if verification_results:
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("<b>RESULTADOS DE VERIFICACIÓN NUTRICIONAL</b>",
                                      ParagraphStyle('SectionTitle', parent=self.styles['Heading2'],
                                                     fontSize=14, textColor=colors.darkblue)))
            elements.append(Spacer(1, 10))

            verification_data = [
                ['Parámetro', 'Objetivo (mg/L)', 'Actual (mg/L)', 'Desviación (%)', 'Estado']]

            for result in verification_results:
                verification_data.append([
                    result.get('parameter', ''),
                    f"{result.get('target_value', 0):.1f}",
                    f"{result.get('actual_value', 0):.1f}",
                    f"{result.get('percentage_deviation', 0):+.1f}%",
                    result.get('status', '')
                ])

            verification_table = Table(verification_data, colWidths=[
                                       1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            verification_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.white, colors.lightgrey]),
            ]))

            elements.append(verification_table)

        # Ionic Balance Analysis
        ionic_balance = calc_results.get('ionic_balance', {})
        if ionic_balance:
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("<b>ANÁLISIS DE BALANCE IÓNICO</b>",
                                      ParagraphStyle('SectionTitle', parent=self.styles['Heading2'],
                                                     fontSize=14, textColor=colors.darkblue)))
            elements.append(Spacer(1, 10))

            balance_data = [
                ['Parámetro', 'Valor', 'Unidad'],
                ['Suma de Cationes',
                    f"{ionic_balance.get('cation_sum', 0):.2f}", 'meq/L'],
                ['Suma de Aniones',
                    f"{ionic_balance.get('anion_sum', 0):.2f}", 'meq/L'],
                ['Diferencia',
                    f"{ionic_balance.get('difference', 0):.2f}", 'meq/L']
            ]

            balance_table = Table(balance_data, colWidths=[
                                  2.5*inch, 1.5*inch, 1*inch])
            balance_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.white, colors.lightgrey]),
                ('TEXTCOLOR', (1, -1), (1, -1),
                 colors.green if ionic_balance.get('is_balanced') == 1 else colors.red),
                ('FONTNAME', (1, -1), (1, -1), 'Helvetica-Bold'),
            ]))

            elements.append(balance_table)

        # Cost Analysis Table
        cost_analysis = calc_results.get('cost_analysis', {})
        if cost_analysis and cost_analysis.get('cost_per_fertilizer'):
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("<b>ANÁLISIS ECONÓMICO</b>",
                                      ParagraphStyle('SectionTitle', parent=self.styles['Heading2'],
                                                     fontSize=14, textColor=colors.darkblue)))
            elements.append(Spacer(1, 10))

            cost_data = [
                ['Fertilizante', 'Costo por 1000L ($)', 'Porcentaje del Total (%)']]

            cost_per_fert = cost_analysis.get('cost_per_fertilizer', {})
            percentage_per_fert = cost_analysis.get(
                'percentage_per_fertilizer', {})

            for fert, cost in cost_per_fert.items():
                if cost > 0:
                    percentage = percentage_per_fert.get(fert, 0)
                    cost_data.append([
                        fert,
                        f"${cost:.3f}",
                        f"{percentage:.1f}%"
                    ])

            # Add total row
            total_cost = cost_analysis.get('total_cost_diluted', 0)
            cost_data.append(['TOTAL', f"${total_cost:.2f}", '100.0%'])

            cost_table = Table(cost_data, colWidths=[3*inch, 2*inch, 2*inch])
            cost_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('BACKGROUND', (0, -1), (-1, -1), colors.lightyellow),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -2), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -2),
                 [colors.white, colors.lightgrey]),
            ]))

            elements.append(cost_table)

        return elements
