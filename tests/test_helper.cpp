// tests/test_helper.cpp
#include <gtest/gtest.h>
#include "../include/core/parsing/input_parser.hpp"
#include "../include/visualization/visualization.hpp"
#include "../include/types/heating_design_structures.hpp"
#include <sstream>
#include <json/json.h>
#include <fstream>
#include "../include/core/parsing/parser/ar_design_parser.hpp"
#include <opencv2/opencv.hpp>

class ParseARDesignTest : public ::testing::Test {
protected:
    // Helper function to create a minimal valid JSON string
    static std::string createMinimalValidJson() {
        return R"({
            "Floor": [{
                "Name": "F1",
                "Num": "1",
                "LevelHeight": 3.0,
                "Construction": {
                    "HouseType": [{
                        "houseName": "House1",
                        "RoomNames": ["LivingRoom", "Bedroom"],
                        "Boundary": [
                            {"X": 0.0, "Y": 0.0, "Z": 0.0},
                            {"X": 10.0, "Y": 0.0, "Z": 0.0},
                            {"X": 10.0, "Y": 10.0, "Z": 0.0}
                        ]
                    }],
                    "Room": [{
                        "Guid": "room-001",
                        "Name": "LivingRoom",
                        "NameType": "Living",
                        "DoorIds": ["door-001"],
                        "BlCreateRoom": 1,
                        "Boundary": [{
                            "StartPoint": {"X": 0.0, "Y": 0.0, "Z": 0.0},
                            "EndPoint": {"X": 5.0, "Y": 0.0, "Z": 0.0},
                            "CurveType": 0
                        }]
                    }]
                }
            }]})";
    }
};

// Test parsing a valid minimal JSON
TEST_F(ParseARDesignTest, ParsesValidJson) {
    std::string json = createMinimalValidJson();
    ARDesign result = iad::parsers::ARDesignParser::parse(json);

    ASSERT_EQ(result.Floor.size(), 1);
    ASSERT_EQ(result.Floor[0].Name, "F1");
    ASSERT_EQ(result.Floor[0].Num, "1");
    ASSERT_DOUBLE_EQ(result.Floor[0].LevelHeight, 3.0);

    // Test HouseType parsing
    const auto &houseTypes = result.Floor[0].construction.houseTypes;
    ASSERT_EQ(houseTypes.size(), 1);
    ASSERT_EQ(houseTypes[0].houseName, "House1");
    ASSERT_EQ(houseTypes[0].RoomNames.size(), 2);
    ASSERT_EQ(houseTypes[0].RoomNames[0], "LivingRoom");
    ASSERT_EQ(houseTypes[0].Boundary.size(), 3);
    ASSERT_DOUBLE_EQ(houseTypes[0].Boundary[0].x, 0.0);

    // Test Room parsing
    const auto &rooms = result.Floor[0].construction.rooms;
    ASSERT_EQ(rooms.size(), 1);
    ASSERT_EQ(rooms[0].Guid, "room-001");
    ASSERT_EQ(rooms[0].Name, "LivingRoom");
    ASSERT_EQ(rooms[0].DoorIds.size(), 1);
    ASSERT_EQ(rooms[0].BlCreateRoom, 1);
}

// Test parsing invalid JSON
TEST_F(ParseARDesignTest, ThrowsOnInvalidJson) {
    std::string invalidJson = "{ invalid json }";
    ASSERT_THROW(iad::parsers::ARDesignParser::parse(invalidJson), std::runtime_error);
}

// Test parsing empty JSON
TEST_F(ParseARDesignTest, HandlesEmptyJson) {
    std::string emptyJson = "{}";
    ARDesign result = iad::parsers::ARDesignParser::parse(emptyJson);
    ASSERT_EQ(result.Floor.size(), 0);
}

// Test parsing JSON with missing optional fields
TEST_F(ParseARDesignTest, HandlesMissingOptionalFields) {
    std::string jsonWithMissingFields = R"({
        "Floor": [{
            "Name": "F1",
            "Num": "1",
            "construction": {
                "houseTypes": []
            }
        }]
    })";

    ARDesign result = iad::parsers::ARDesignParser::parse(jsonWithMissingFields);
    ASSERT_EQ(result.Floor.size(), 1);
    ASSERT_EQ(result.Floor[0].Name, "F1");
    ASSERT_DOUBLE_EQ(result.Floor[0].LevelHeight, 0.0); // Default value
    ASSERT_EQ(result.Floor[0].construction.houseTypes.size(), 0);
}


// Test parsing actual ARDesign-min.json file
TEST_F(ParseARDesignTest, ParsesActualARDesignFile) {
    // 获取当前工作目录（帮助调试）
    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));
    // Read file content
    // std::string filePath = "../../example/ARDesign-min.json";
    std::string filePath = "../example/ARDesign2.json";
    // std::string filePath = "../../example/ARDesign2.json";
    // std::string filePath = "../example/ARDesign01.json";
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Current working directory: " << cwd << std::endl;
        std::cerr << "Attempted to open file at: " << filePath << std::endl;
        FAIL() << "Failed to open ARDesign-min.json";
    }
    ASSERT_TRUE(file.is_open()) << "Failed to open ARDesign-min.json";
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    
    // Parse JSON
    ARDesign result = iad::parsers::ARDesignParser::parse(json);

    // Test basic structure
    // ASSERT_EQ(result.Floor.size(), 1);
    ASSERT_GT(result.Floor.size(), 0);
    ASSERT_EQ(result.Floor[0].Name, "-1");
    ASSERT_EQ(result.Floor[0].Num, "-1");
    ASSERT_DOUBLE_EQ(result.Floor[0].LevelHeight, 3750.0);

    // Test Construction components
    const auto& construction = result.Floor[0].construction;

    // print all construction components
    std::cout << "construction: " << construction.rooms.size() << std::endl;
    std::cout << "jcws: " << construction.jcws.size() << std::endl;
    std::cout << "door: " << construction.door.size() << std::endl;
    
    // Test Room
    ASSERT_FALSE(construction.rooms.empty());
    const auto& room = construction.rooms[0];
    ASSERT_FALSE(room.Boundary.empty());
    ASSERT_EQ(room.BlCreateRoom, 1);

    // Create a temporary file path for the output image
    std::string outputPath = "outputs/test_ar_design.png";

    // Call the drawing function
    iad::drawARDesign(result, outputPath);

}

TEST_F(ParseARDesignTest, ParsesActualARDesignFile01) {
    // 获取当前工作目录（帮助调试）
    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));
    // Read file content
    // std::string filePath = "../../example/ARDesign-min.json";
    // std::string filePath = "../example/ARDesign2.json";
    // std::string filePath = "../example/ARDesign02.json";
    std::string filePath = "../../example/ARDesign02.json";
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Current working directory: " << cwd << std::endl;
        std::cerr << "Attempted to open file at: " << filePath << std::endl;
        FAIL() << "Failed to open ARDesign-min.json";
    }
    ASSERT_TRUE(file.is_open()) << "Failed to open ARDesign-min.json";
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    
    // Parse JSON
    ARDesign result = iad::parsers::ARDesignParser::parse(json);

    // Test Construction components
    const auto& construction = result.Floor[0].construction;

    // print all construction components
    std::cout << "construction: " << construction.rooms.size() << std::endl;
    std::cout << "jcws: " << construction.jcws.size() << std::endl;
    std::cout << "door: " << construction.door.size() << std::endl;
    

    // Create a temporary file path for the output image
    std::string outputPath = "outputs/test_ar_design.png";

    // Call the drawing function
    iad::drawARDesign(result, outputPath);

}

TEST_F(ParseARDesignTest, DrawARDesignTest) {
    // Create test ARDesign with multiple floors
    ARDesign testDesign;
    
    // Create two test floors
    for (const auto& floorNum : {"1", "2"}) {
        Floor floor;
        floor.Name = floorNum;
        floor.Num = floorNum;
        floor.LevelHeight = 3000.0;

        // Add a room with rectangular boundary
        Room room;
        room.Name = "TestRoom";
        room.Guid = "room-001";
        room.BlCreateRoom = 1;

        // Create a rectangular room boundary
        CurveInfo curve1, curve2, curve3, curve4;
        
        // Bottom line
        curve1.StartPoint = {0.0, 0.0, 0.0};
        curve1.EndPoint = {5000.0, 0.0, 0.0};
        curve1.CurveType = 0;

        // Right line
        curve2.StartPoint = {5000.0, 0.0, 0.0};
        curve2.EndPoint = {5000.0, 4000.0, 0.0};
        curve2.CurveType = 0;

        // Top line
        curve3.StartPoint = {5000.0, 4000.0, 0.0};
        curve3.EndPoint = {0.0, 4000.0, 0.0};
        curve3.CurveType = 0;

        // Left line
        curve4.StartPoint = {0.0, 4000.0, 0.0};
        curve4.EndPoint = {0.0, 0.0, 0.0};
        curve4.CurveType = 0;

        room.Boundary = {curve1, curve2, curve3, curve4};
        floor.construction.rooms.push_back(room);

        testDesign.Floor.push_back(floor);
    }

    // Create a temporary file path for the output image
    std::string outputPath = "test_ar_design.png";

    // Call the drawing function
    iad::drawARDesign(testDesign, outputPath);

    // Verify that the files were created for each floor
    for (const auto& floorNum : {"1", "2"}) {
        std::string expectedPath = "outputs/test_ar_design_floor_" + std::string(floorNum) + ".png";
        std::ifstream file(expectedPath);
        ASSERT_TRUE(file.good()) << "Image file was not created for floor " << floorNum;
        file.close();
        
        // Clean up the test file
        std::remove(expectedPath.c_str());
    }
}
