-- Data Preprocessing SQL Script for Environmental Data

-- Step 1: Handling Missing Values
-- Impute missing values for selected columns using the mean

-- Calculate mean values
WITH Means AS (
    SELECT
        AVG(PM2.5) AS Mean_PM25,
        AVG(Temperature) AS Mean_Temperature,
        AVG(Humidity) AS Mean_Humidity,
        AVG(Soil_Moisture) AS Mean_SoilMoisture,
        AVG(Biodiversity_Index) AS Mean_BiodiversityIndex,
        AVG(Nutrient_Level) AS Mean_NutrientLevel,
        AVG(Water_Quality) AS Mean_WaterQuality,
        AVG(Air_Quality_Index) AS Mean_AirQualityIndex,
        AVG(Pollution_Level) AS Mean_PollutionLevel,
        AVG(Soil_pH) AS Mean_SoilPh,
        AVG(Dissolved_Oxygen) AS Mean_DissolvedOxygen,
        AVG(Chemical_Oxygen_Demand) AS Mean_ChemicalOxygenDemand,
        AVG(Biochemical_Oxygen_Demand) AS Mean_BiochemicalOxygenDemand,
        AVG(Total_Dissolved_Solids) AS Mean_TotalDissolvedSolids
    FROM environmental_data
    WHERE PM2.5 IS NOT NULL
      AND Temperature IS NOT NULL
      AND Humidity IS NOT NULL
      AND Soil_Moisture IS NOT NULL
      AND Biodiversity_Index IS NOT NULL
      AND Nutrient_Level IS NOT NULL
      AND Water_Quality IS NOT NULL
      AND Air_Quality_Index IS NOT NULL
      AND Pollution_Level IS NOT NULL
      AND Soil_pH IS NOT NULL
      AND Dissolved_Oxygen IS NOT NULL
      AND Chemical_Oxygen_Demand IS NOT NULL
      AND Biochemical_Oxygen_Demand IS NOT NULL
      AND Total_Dissolved_Solids IS NOT NULL
)

-- Update table to fill missing values
UPDATE environmental_data
SET
    PM2.5 = (SELECT Mean_PM25 FROM Means) WHERE PM2.5 IS NULL,
    Temperature = (SELECT Mean_Temperature FROM Means) WHERE Temperature IS NULL,
    Humidity = (SELECT Mean_Humidity FROM Means) WHERE Humidity IS NULL,
    Soil_Moisture = (SELECT Mean_SoilMoisture FROM Means) WHERE Soil_Moisture IS NULL,
    Biodiversity_Index = (SELECT Mean_BiodiversityIndex FROM Means) WHERE Biodiversity_Index IS NULL,
    Nutrient_Level = (SELECT Mean_NutrientLevel FROM Means) WHERE Nutrient_Level IS NULL,
    Water_Quality = (SELECT Mean_WaterQuality FROM Means) WHERE Water_Quality IS NULL,
    Air_Quality_Index = (SELECT Mean_AirQualityIndex FROM Means) WHERE Air_Quality_Index IS NULL,
    Pollution_Level = (SELECT Mean_PollutionLevel FROM Means) WHERE Pollution_Level IS NULL,
    Soil_pH = (SELECT Mean_SoilPh FROM Means) WHERE Soil_pH IS NULL,
    Dissolved_Oxygen = (SELECT Mean_DissolvedOxygen FROM Means) WHERE Dissolved_Oxygen IS NULL,
    Chemical_Oxygen_Demand = (SELECT Mean_ChemicalOxygenDemand FROM Means) WHERE Chemical_Oxygen_Demand IS NULL,
    Biochemical_Oxygen_Demand = (SELECT Mean_BiochemicalOxygenDemand FROM Means) WHERE Biochemical_Oxygen_Demand IS NULL,
    Total_Dissolved_Solids = (SELECT Mean_TotalDissolvedSolids FROM Means) WHERE Total_Dissolved_Solids IS NULL;


-- Step 2: Normalization/Scaling
-- Normalize numerical columns to a range between 0 and 1

-- Calculate min and max values for each column
WITH MinMax AS (
    SELECT
        MIN(PM2.5) AS Min_PM25, MAX(PM2.5) AS Max_PM25,
        MIN(Temperature) AS Min_Temperature, MAX(Temperature) AS Max_Temperature,
        MIN(Humidity) AS Min_Humidity, MAX(Humidity) AS Max_Humidity,
        MIN(Soil_Moisture) AS Min_SoilMoisture, MAX(Soil_Moisture) AS Max_SoilMoisture,
        MIN(Biodiversity_Index) AS Min_BiodiversityIndex, MAX(Biodiversity_Index) AS Max_BiodiversityIndex,
        MIN(Nutrient_Level) AS Min_NutrientLevel, MAX(Nutrient_Level) AS Max_NutrientLevel,
        MIN(Water_Quality) AS Min_WaterQuality, MAX(Water_Quality) AS Max_WaterQuality,
        MIN(Air_Quality_Index) AS Min_AirQualityIndex, MAX(Air_Quality_Index) AS Max_AirQualityIndex,
        MIN(Pollution_Level) AS Min_PollutionLevel, MAX(Pollution_Level) AS Max_PollutionLevel,
        MIN(Soil_pH) AS Min_SoilPh, MAX(Soil_pH) AS Max_SoilPh,
        MIN(Dissolved_Oxygen) AS Min_DissolvedOxygen, MAX(Dissolved_Oxygen) AS Max_DissolvedOxygen,
        MIN(Chemical_Oxygen_Demand) AS Min_ChemicalOxygenDemand, MAX(Chemical_Oxygen_Demand) AS Max_ChemicalOxygenDemand,
        MIN(Biochemical_Oxygen_Demand) AS Min_BiochemicalOxygenDemand, MAX(Biochemical_Oxygen_Demand) AS Max_BiochemicalOxygenDemand,
        MIN(Total_Dissolved_Solids) AS Min_TotalDissolvedSolids, MAX(Total_Dissolved_Solids) AS Max_TotalDissolvedSolids
    FROM environmental_data
)

-- Update table with normalized values
UPDATE environmental_data
SET
    PM2.5 = (PM2.5 - (SELECT Min_PM25 FROM MinMax)) / ((SELECT Max_PM25 FROM MinMax) - (SELECT Min_PM25 FROM MinMax)),
    Temperature = (Temperature - (SELECT Min_Temperature FROM MinMax)) / ((SELECT Max_Temperature FROM MinMax) - (SELECT Min_Temperature FROM MinMax)),
    Humidity = (Humidity - (SELECT Min_Humidity FROM MinMax)) / ((SELECT Max_Humidity FROM MinMax) - (SELECT Min_Humidity FROM MinMax)),
    Soil_Moisture = (Soil_Moisture - (SELECT Min_SoilMoisture FROM MinMax)) / ((SELECT Max_SoilMoisture FROM MinMax) - (SELECT Min_SoilMoisture FROM MinMax)),
    Biodiversity_Index = (Biodiversity_Index - (SELECT Min_BiodiversityIndex FROM MinMax)) / ((SELECT Max_BiodiversityIndex FROM MinMax) - (SELECT Min_BiodiversityIndex FROM MinMax)),
    Nutrient_Level = (Nutrient_Level - (SELECT Min_NutrientLevel FROM MinMax)) / ((SELECT Max_NutrientLevel FROM MinMax) - (SELECT Min_NutrientLevel FROM MinMax)),
    Water_Quality = (Water_Quality - (SELECT Min_WaterQuality FROM MinMax)) / ((SELECT Max_WaterQuality FROM MinMax) - (SELECT Min_WaterQuality FROM MinMax)),
    Air_Quality_Index = (Air_Quality_Index - (SELECT Min_AirQualityIndex FROM MinMax)) / ((SELECT Max_AirQualityIndex FROM MinMax) - (SELECT Min_AirQualityIndex FROM MinMax)),
    Pollution_Level = (Pollution_Level - (SELECT Min_PollutionLevel FROM MinMax)) / ((SELECT Max_PollutionLevel FROM MinMax) - (SELECT Min_PollutionLevel FROM MinMax)),
    Soil_pH = (Soil_pH - (SELECT Min_SoilPh FROM MinMax)) / ((SELECT Max_SoilPh FROM MinMax) - (SELECT Min_SoilPh FROM MinMax)),
    Dissolved_Oxygen = (Dissolved_Oxygen - (SELECT Min_DissolvedOxygen FROM MinMax)) / ((SELECT Max_DissolvedOxygen FROM MinMax) - (SELECT Min_DissolvedOxygen FROM MinMax)),
    Chemical_Oxygen_Demand = (Chemical_Oxygen_Demand - (SELECT Min_ChemicalOxygenDemand FROM MinMax)) / ((SELECT Max_ChemicalOxygenDemand FROM MinMax) - (SELECT Min_ChemicalOxygenDemand FROM MinMax)),
    Biochemical_Oxygen_Demand = (Biochemical_Oxygen_Demand - (SELECT Min_BiochemicalOxygenDemand FROM MinMax)) / ((SELECT Max_BiochemicalOxygenDemand FROM MinMax) - (SELECT Min_BiochemicalOxygenDemand FROM MinMax)),
    Total_Dissolved_Solids = (Total_Dissolved_Solids - (SELECT Min_TotalDissolvedSolids FROM MinMax)) / ((SELECT Max_TotalDissolvedSolids FROM MinMax) - (SELECT Min_TotalDissolvedSolids FROM MinMax));


-- Step 3: Encoding Categorical Data
-- Map unique labels of Ecological_Health_Label to integers

-- Create label mapping table
CREATE TABLE LabelMapping AS
SELECT DISTINCT Ecological_Health_Label, 
       ROW_NUMBER() OVER (ORDER BY Ecological_Health_Label) AS LabelCode
FROM environmental_data;

-- Update original table with label codes
UPDATE environmental_data
SET Ecological_Health_Label = (
    SELECT LabelCode
    FROM LabelMapping
    WHERE LabelMapping.Ecological_Health_Label = environmental_data.Ecological_Health_Label
);


-- Step 4: Date/Time Transformation
-- Extract useful features from Timestamp (e
