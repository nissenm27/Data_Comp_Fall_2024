#include <vector>             // For using std::vector containers
#include <string>             // For using std::string
#include <fstream>            // For file operations (std::ifstream)
#include <sstream>            // For string stream operations (std::istringstream)
#include <algorithm>          // For algorithms like std::min
#include <iostream>           // For debugging outputs if needed

struct EnvironmentalData {
    std::string timestamp;
    double pm25, temperature, humidity, soilMoisture, biodiversityIndex, nutrientLevel;
    double waterQuality, airQualityIndex, pollutionLevel, soilPh, dissolvedOxygen;
    double chemicalOxygenDemand, biochemicalOxygenDemand, totalDissolvedSolids;
    int ecologicalHealthLabel;  // Assuming it's categorical
};

// Function to load data from a CSV file
std::vector<EnvironmentalData> loadCSV(const std::string &filename) {
    std::ifstream file(filename);          // Open the file with the given filename
    if (!file.is_open()) {                 // Check if the file was successfully opened
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};                         // Return an empty vector if file opening failed
    }
    
    std::vector<EnvironmentalData> data;   // Vector to store the loaded data
    std::string line;
    
    while (std::getline(file, line)) {     // Read each line from the file
        std::istringstream ss(line);       // Use a string stream to parse the line
        EnvironmentalData entry;           // Create an instance of EnvironmentalData to hold parsed data
        
        // Assuming the CSV format aligns with the struct fields in the exact order
        // For example: timestamp, pm25, temperature, etc., separated by commas
        std::getline(ss, entry.timestamp, ',');  // Parse each field and store in `entry`
        ss >> entry.pm25;               ss.ignore(1);   // Parse pm25, ignore comma
        ss >> entry.temperature;        ss.ignore(1);   // Parse temperature, ignore comma
        ss >> entry.humidity;           ss.ignore(1);   // Parse humidity, ignore comma
        ss >> entry.soilMoisture;       ss.ignore(1);   // Parse soilMoisture, ignore comma
        ss >> entry.biodiversityIndex;  ss.ignore(1);   // Parse biodiversityIndex, ignore comma
        ss >> entry.nutrientLevel;      ss.ignore(1);   // Parse nutrientLevel, ignore comma
        ss >> entry.waterQuality;       ss.ignore(1);   // Parse waterQuality, ignore comma
        ss >> entry.airQualityIndex;    ss.ignore(1);   // Parse airQualityIndex, ignore comma
        ss >> entry.pollutionLevel;     ss.ignore(1);   // Parse pollutionLevel, ignore comma
        ss >> entry.soilPh;             ss.ignore(1);   // Parse soilPh, ignore comma
        ss >> entry.dissolvedOxygen;    ss.ignore(1);   // Parse dissolvedOxygen, ignore comma
        ss >> entry.chemicalOxygenDemand; ss.ignore(1); // Parse chemicalOxygenDemand, ignore comma
        ss >> entry.biochemicalOxygenDemand; ss.ignore(1); // Parse biochemicalOxygenDemand, ignore comma
        ss >> entry.totalDissolvedSolids; ss.ignore(1); // Parse totalDissolvedSolids, ignore comma
        ss >> entry.ecologicalHealthLabel;             // Parse ecologicalHealthLabel
        
        data.push_back(entry);            // Add the populated entry to the data vector
    }
    
    return data;                           // Return the vector containing all data entries
}

// Function to create batches of EnvironmentalData for neural network training
std::vector<std::vector<EnvironmentalData>> createBatches(
    const std::vector<EnvironmentalData>& data, int batchSize) {
    std::vector<std::vector<EnvironmentalData>> batches;   // Vector to store batches
    
    for (size_t i = 0; i < data.size(); i += batchSize) {
        // Create a batch from the range [i, i + batchSize) or to the end if size is smaller
        std::vector<EnvironmentalData> batch(data.begin() + i,
                                             data.begin() + std::min(i + batchSize, data.size()));
        batches.push_back(batch);                          // Add the batch to the batches vector
    }
    
    return batches;                                        // Return the vector of batches
}
