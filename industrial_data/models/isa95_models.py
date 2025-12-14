"""
ISA-95 Data Models for Manufacturing Operations

ISA-95 (ANSI/ISA-95 or IEC/ISO 62264) is the international standard for
integrating enterprise and control systems.

This module provides data models compliant with ISA-95 Part 2
(Object Model Attributes).

Hierarchy Levels:
- Level 4: Business Planning & Logistics (ERP)
- Level 3: Manufacturing Operations Management (MES)
- Level 2: Supervisory Control (SCADA)
- Level 1: Control (PLC, DCS)
- Level 0: Physical Process (Sensors, Actuators)

SwarmBrain operates primarily at Level 2-3 interface.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ============================================================================
# Equipment Models (ISA-95 Part 2, Section 4)
# ============================================================================

class EquipmentLevel(Enum):
    """Equipment hierarchy levels."""
    ENTERPRISE = "enterprise"
    SITE = "site"
    AREA = "area"
    PROCESS_CELL = "process_cell"
    UNIT = "unit"
    PRODUCTION_LINE = "production_line"
    WORK_CENTER = "work_center"
    WORK_UNIT = "work_unit"
    EQUIPMENT_MODULE = "equipment_module"


@dataclass
class Equipment:
    """
    Equipment definition (ISA-95 Part 2, Section 4.2).

    Represents physical equipment in the manufacturing facility.
    """
    equipment_id: str
    equipment_name: str
    equipment_level: EquipmentLevel
    description: str = ""
    parent_equipment_id: Optional[str] = None
    equipment_class_id: Optional[str] = None
    vendor: Optional[str] = None
    model_number: Optional[str] = None
    serial_number: Optional[str] = None
    install_date: Optional[datetime] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)


@dataclass
class EquipmentCapability:
    """
    Equipment capability test specification.

    Defines what the equipment can do (capacity, rates, tolerances).
    """
    capability_id: str
    capability_type: str
    description: str
    capacity: Optional[float] = None
    capacity_unit: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Personnel Models (ISA-95 Part 2, Section 5)
# ============================================================================

class PersonnelClassType(Enum):
    """Types of personnel classifications."""
    OPERATOR = "operator"
    SUPERVISOR = "supervisor"
    ENGINEER = "engineer"
    TECHNICIAN = "technician"
    QUALITY_INSPECTOR = "quality_inspector"
    MAINTENANCE = "maintenance"


@dataclass
class Personnel:
    """Personnel definition (ISA-95 Part 2, Section 5.2)."""
    person_id: str
    person_name: str
    description: str = ""
    personnel_class: Optional[PersonnelClassType] = None
    qualifications: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Material Models (ISA-95 Part 2, Section 6)
# ============================================================================

class MaterialDefinitionLevel(Enum):
    """Material hierarchy levels."""
    FINISHED_PRODUCT = "finished_product"
    INTERMEDIATE = "intermediate"
    RAW_MATERIAL = "raw_material"
    PACKAGING = "packaging"
    TOOL = "tool"
    CONSUMABLE = "consumable"


@dataclass
class MaterialDefinition:
    """Material definition (ISA-95 Part 2, Section 6.2)."""
    material_id: str
    material_name: str
    description: str = ""
    material_level: Optional[MaterialDefinitionLevel] = None
    material_class_id: Optional[str] = None
    assembly_type: Optional[str] = None  # "assembly", "component", "bulk"
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaterialLot:
    """Material lot (specific instance of material)."""
    lot_id: str
    material_id: str
    quantity: float
    unit: str
    status: str  # "available", "quarantine", "consumed", "scrapped"
    location: Optional[str] = None
    manufactured_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    properties: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Process Segment Models (ISA-95 Part 2, Section 7)
# ============================================================================

class ProcessSegmentType(Enum):
    """Types of process segments."""
    PRODUCTION = "production"
    MAINTENANCE = "maintenance"
    QUALITY = "quality"
    INVENTORY = "inventory"
    MIXED = "mixed"


@dataclass
class ProcessSegment:
    """
    Process segment definition (ISA-95 Part 2, Section 7.2).

    Defines a logical grouping of operations.
    """
    segment_id: str
    segment_name: str
    description: str = ""
    segment_type: Optional[ProcessSegmentType] = None
    duration: Optional[float] = None  # minutes
    duration_unit: str = "minutes"
    operations: List[str] = field(default_factory=list)  # Operation IDs
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationDefinition:
    """Operation within a process segment."""
    operation_id: str
    operation_name: str
    description: str = ""
    duration: Optional[float] = None
    equipment_requirements: List[str] = field(default_factory=list)
    material_requirements: List[str] = field(default_factory=list)
    personnel_requirements: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Production Schedule Models (ISA-95 Part 3)
# ============================================================================

class ProductionRequestState(Enum):
    """Production request states."""
    FORECAST = "forecast"
    PLANNED = "planned"
    RELEASED = "released"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    HELD = "held"
    CLOSED = "closed"


@dataclass
class ProductionRequest:
    """
    Production request (ISA-95 Part 3).

    A request to produce material.
    """
    request_id: str
    description: str
    state: ProductionRequestState
    segment_id: str  # Process segment to execute
    start_time: datetime
    end_time: datetime
    priority: int = 50  # 0-100
    material_produced: Optional[str] = None
    quantity_requested: Optional[float] = None
    quantity_unit: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductionSchedule:
    """Production schedule (collection of production requests)."""
    schedule_id: str
    schedule_name: str
    start_time: datetime
    end_time: datetime
    requests: List[ProductionRequest] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Production Performance Models (ISA-95 Part 3)
# ============================================================================

@dataclass
class ProductionPerformance:
    """
    Production performance (actual production results).

    ISA-95 Part 3, Section 3.3
    """
    performance_id: str
    production_request_id: str
    start_time: datetime
    end_time: datetime
    quantity_produced: float
    quantity_unit: str
    equipment_used: List[str] = field(default_factory=list)
    material_consumed: List[str] = field(default_factory=list)
    personnel: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Resource Mapping Helper
# ============================================================================

class ISA95ResourceMapper:
    """
    Maps SwarmBrain resources to ISA-95 model.

    Provides bidirectional mapping between SwarmBrain robot/task
    concepts and ISA-95 manufacturing concepts.
    """

    @staticmethod
    def robot_to_equipment(robot_id: str, capabilities: List[str]) -> Equipment:
        """Map SwarmBrain robot to ISA-95 Equipment."""
        return Equipment(
            equipment_id=robot_id,
            equipment_name=f"Robot {robot_id}",
            equipment_level=EquipmentLevel.WORK_UNIT,
            description=f"Humanoid robot for collaborative tasks",
            capabilities=capabilities,
            properties={
                'type': 'collaborative_robot',
                'vendor': 'SwarmBrain',
                'autonomy_level': 'semi-autonomous'
            }
        )

    @staticmethod
    def work_order_to_production_request(
        work_order_id: str,
        product_id: str,
        quantity: float,
        unit: str,
        start_time: datetime,
        end_time: datetime,
        priority: int = 50
    ) -> ProductionRequest:
        """Map MES work order to ISA-95 Production Request."""
        return ProductionRequest(
            request_id=work_order_id,
            description=f"Produce {quantity} {unit} of {product_id}",
            state=ProductionRequestState.PLANNED,
            segment_id=f"segment_{work_order_id}",
            start_time=start_time,
            end_time=end_time,
            priority=priority,
            material_produced=product_id,
            quantity_requested=quantity,
            quantity_unit=unit
        )

    @staticmethod
    def skill_to_operation(
        skill_name: str,
        duration: Optional[float] = None
    ) -> OperationDefinition:
        """Map SwarmBrain skill to ISA-95 Operation."""
        return OperationDefinition(
            operation_id=f"op_{skill_name}",
            operation_name=skill_name.replace('_', ' ').title(),
            description=f"Robot skill: {skill_name}",
            duration=duration,
            equipment_requirements=['collaborative_robot'],
            properties={'skill_type': skill_name}
        )
