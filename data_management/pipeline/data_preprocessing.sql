CREATE DATABASE ecological_data;
USE ecological_data;

-- Step 1: Create a table for the raw data if not already existing
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

-- Step 2: Load data into the table
-- Make sure 'ecological_health_dataset.csv' is accessible to the MySQL server.
-- Replace 'path/to/ecological_health_dataset.csv' with the actual file path.
LOAD DATA INFILE '/Users/mattn/Documents/Fall_24:25_Sem/CMDA_Data_Competition/data_management/pipeline/ecological_health_dataset.csv'
INTO TABLE environmental_data
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- Step 3: Transformations
-- Extract Year, Month, Day, Hour, Minute from the Timestamp
-- Map categorical columns (Pollution_Level, Ecological_Health_Label) to numeric values

-- Create a new table to store the transformed data
CREATE TABLE transformed_environmental_data AS
SELECT 
    -- Extract date and time components from the Timestamp
    YEAR(Timestamp) AS Year,
    MONTH(Timestamp) AS Month,
    DAY(Timestamp) AS Day,
    HOUR(Timestamp) AS Hour,
    MINUTE(Timestamp) AS Minute,
    
    -- Numeric features (retain as they are)
    PM25,
    Temperature,
    Humidity,
    Soil_Moisture,
    Biodiversity_Index,
    Nutrient_Level,
    Water_Quality,
    Air_Quality_Index,
    Soil_pH,
    Dissolved_Oxygen,
    Chemical_Oxygen_Demand,
    Biochemical_Oxygen_Demand,
    Total_Dissolved_Solids,
    
    -- Map Pollution_Level to numerical values
    CASE 
        WHEN Pollution_Level = 'Low' THEN 0
        WHEN Pollution_Level = 'Moderate' THEN 1
        WHEN Pollution_Level = 'High' THEN 2
        ELSE NULL
    END AS Pollution_Level,
    
    -- Map Ecological_Health_Label to numerical values
    CASE 
        WHEN Ecological_Health_Label = 'Ecologically Healthy' THEN 4
        WHEN Ecological_Health_Label = 'Ecologically Stable' THEN 3
        WHEN Ecological_Health_Label = 'Ecologically Critical' THEN 2
        WHEN Ecological_Health_Label = 'Ecologically Degraded' THEN 1
        ELSE NULL
    END AS Ecological_Health_Label
    
FROM environmental_data;

-- Step 4: Handle Missing Values
-- Impute missing values with the mean for each column

-- Calculate and update each column individually
UPDATE transformed_environmental_data
SET 
    PM25 = IFNULL(PM25, (SELECT AVG(PM25) FROM environmental_data)),
    Temperature = IFNULL(Temperature, (SELECT AVG(Temperature) FROM environmental_data)),
    Humidity = IFNULL(Humidity, (SELECT AVG(Humidity) FROM environmental_data)),
    Soil_Moisture = IFNULL(Soil_Moisture, (SELECT AVG(Soil_Moisture) FROM environmental_data)),
    Biodiversity_Index = IFNULL(Biodiversity_Index, (SELECT AVG(Biodiversity_Index) FROM environmental_data)),
    Nutrient_Level = IFNULL(Nutrient_Level, (SELECT AVG(Nutrient_Level) FROM environmental_data)),
    Water_Quality = IFNULL(Water_Quality, (SELECT AVG(Water_Quality) FROM environmental_data)),
    Air_Quality_Index = IFNULL(Air_Quality_Index, (SELECT AVG(Air_Quality_Index) FROM environmental_data)),
    -- Correct Pollution_Level usage to refer to `environmental_data`
    Pollution_Level = IFNULL(Pollution_Level, (SELECT AVG(CASE 
        WHEN Pollution_Level = 'Low' THEN 0 
        WHEN Pollution_Level = 'Moderate' THEN 1 
        WHEN Pollution_Level = 'High' THEN 2
        ELSE NULL END) 
    FROM environmental_data)),
    Soil_pH = IFNULL(Soil_pH, (SELECT AVG(Soil_pH) FROM environmental_data)),
    Dissolved_Oxygen = IFNULL(Dissolved_Oxygen, (SELECT AVG(Dissolved_Oxygen) FROM environmental_data)),
    Chemical_Oxygen_Demand = IFNULL(Chemical_Oxygen_Demand, (SELECT AVG(Chemical_Oxygen_Demand) FROM environmental_data)),
    Biochemical_Oxygen_Demand = IFNULL(Biochemical_Oxygen_Demand, (SELECT AVG(Biochemical_Oxygen_Demand) FROM environmental_data)),
    Total_Dissolved_Solids = IFNULL(Total_Dissolved_Solids, (SELECT AVG(Total_Dissolved_Solids) FROM environmental_data));
