-- Step 1: Create Database and Base Table
CREATE DATABASE IF NOT EXISTS ecological_data;
USE ecological_data;

-- Create the base table for raw data import
CREATE TABLE IF NOT EXISTS environmental_data (
    Timestamp DATETIME,
    PM25 FLOAT,
    Temperature FLOAT,
    Humidity FLOAT,
    Soil_Moisture FLOAT,
    Biodiversity_Index INT,
    Nutrient_Level INT,
    Water_Quality INT,
    Air_Quality_Index FLOAT,
    Pollution_Level VARCHAR(20),
    Soil_pH FLOAT,
    Dissolved_Oxygen FLOAT,
    Chemical_Oxygen_Demand FLOAT,
    Biochemical_Oxygen_Demand FLOAT,
    Total_Dissolved_Solids FLOAT,
    Ecological_Health_Label VARCHAR(50)
);

-- Step 2: Load Data from CSV into environmental_data
-- Make sure the file path is accessible to the MySQL server.
LOAD DATA INFILE '/Users/mattn/Documents/Fall_24:25_Sem/CMDA_Data_Competition/data_management/pipeline/ecological_health_dataset.csv'
INTO TABLE environmental_data
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- Step 3: Create and Transform Data in transformed_environmental_data
CREATE TABLE transformed_environmental_data AS
SELECT 
    -- Extract date and time components from the Timestamp
    YEAR(Timestamp) AS Year,
    MONTH(Timestamp) AS Month,
    DAY(Timestamp) AS Day,
    HOUR(Timestamp) AS Hour,
    MINUTE(Timestamp) AS Minute,
    
    -- Use COALESCE to handle missing values directly in the transformation
    COALESCE(PM25, (SELECT AVG(PM25) FROM environmental_data)) AS PM25,
    COALESCE(Temperature, (SELECT AVG(Temperature) FROM environmental_data)) AS Temperature,
    COALESCE(Humidity, (SELECT AVG(Humidity) FROM environmental_data)) AS Humidity,
    COALESCE(Soil_Moisture, (SELECT AVG(Soil_Moisture) FROM environmental_data)) AS Soil_Moisture,
    COALESCE(Biodiversity_Index, (SELECT AVG(Biodiversity_Index) FROM environmental_data)) AS Biodiversity_Index,
    COALESCE(Nutrient_Level, (SELECT AVG(Nutrient_Level) FROM environmental_data)) AS Nutrient_Level,
    COALESCE(Water_Quality, (SELECT AVG(Water_Quality) FROM environmental_data)) AS Water_Quality,
    COALESCE(Air_Quality_Index, (SELECT AVG(Air_Quality_Index) FROM environmental_data)) AS Air_Quality_Index,
    COALESCE(Soil_pH, (SELECT AVG(Soil_pH) FROM environmental_data)) AS Soil_pH,
    COALESCE(Dissolved_Oxygen, (SELECT AVG(Dissolved_Oxygen) FROM environmental_data)) AS Dissolved_Oxygen,
    COALESCE(Chemical_Oxygen_Demand, (SELECT AVG(Chemical_Oxygen_Demand) FROM environmental_data)) AS Chemical_Oxygen_Demand,
    COALESCE(Biochemical_Oxygen_Demand, (SELECT AVG(Biochemical_Oxygen_Demand) FROM environmental_data)) AS Biochemical_Oxygen_Demand,
    COALESCE(Total_Dissolved_Solids, (SELECT AVG(Total_Dissolved_Solids) FROM environmental_data)) AS Total_Dissolved_Solids,
    
    -- Map `Pollution_Level` to numerical values
    CASE 
        WHEN Pollution_Level = 'Low' THEN 0
        WHEN Pollution_Level = 'Moderate' THEN 1
        WHEN Pollution_Level = 'High' THEN 2
        ELSE NULL
    END AS Pollution_Level,
    
    -- Map `Ecological_Health_Label` to numerical values
    CASE 
        WHEN Ecological_Health_Label = 'Ecologically Healthy' THEN 4
        WHEN Ecological_Health_Label = 'Ecologically Stable' THEN 3
        WHEN Ecological_Health_Label = 'Ecologically Critical' THEN 2
        WHEN Ecological_Health_Label = 'Ecologically Degraded' THEN 1
        ELSE NULL
    END AS Ecological_Health_Label
    
FROM environmental_data;
