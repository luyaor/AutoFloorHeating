// tests/test_helper.cpp
#include <gtest/gtest.h>
#include "../include/helper.hpp"
#include "../include/types/heating_design_structures.hpp"
#include <sstream>
#include <json/json.h>
#include <fstream>

class ParseARDesignTest : public ::testing::Test {
protected:
    // Helper function to create a minimal valid JSON string
    static std::string createMinimalValidJson() {
        return R"({
            "Floor": [{
                "Name": "F1",
                "Num": "1",
                "LevelHeight": 3.0,
                "construction": {
                    "houseTypes": [{
                        "houseName": "House1",
                        "RoomNames": ["LivingRoom", "Bedroom"],
                        "Boundary": [
                            {"X": 0.0, "Y": 0.0, "Z": 0.0},
                            {"X": 10.0, "Y": 0.0, "Z": 0.0},
                            {"X": 10.0, "Y": 10.0, "Z": 0.0}
                        ]
                    }],
                    "rooms": [{
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
    ARDesign result = parseARDesign(json);

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
    ASSERT_THROW(parseARDesign(invalidJson), std::runtime_error);
}

// Test parsing empty JSON
TEST_F(ParseARDesignTest, HandlesEmptyJson) {
    std::string emptyJson = "{}";
    ARDesign result = parseARDesign(emptyJson);
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

    ARDesign result = parseARDesign(jsonWithMissingFields);
    ASSERT_EQ(result.Floor.size(), 1);
    ASSERT_EQ(result.Floor[0].Name, "F1");
    ASSERT_DOUBLE_EQ(result.Floor[0].LevelHeight, 0.0); // Default value
    ASSERT_EQ(result.Floor[0].construction.houseTypes.size(), 0);
}

// Test parsing complete floor with all fields
TEST_F(ParseARDesignTest, ParsesCompleteFloor) {
    std::string completeJson = R"({
        "Floor": [{
            "Name": "F1",
            "Num": "1",
            "LevelHeight": 3.0,
            "construction": {
                "houseTypes": [{
                    "houseName": "House1",
                    "RoomNames": ["LivingRoom"],
                    "Boundary": [
                        {"X": 1.0, "Y": 2.0, "Z": 3.0}
                    ]
                }],
                "rooms": [{
                    "Guid": "room-001",
                    "Name": "LivingRoom",
                    "NameType": "Living",
                    "DoorIds": ["door-001"],
                    "JCWGuidNames": ["jcw-001"],
                    "WallNames": ["wall-001"],
                    "BlCreateRoom": 1,
                    "Boundary": [{
                        "StartPoint": {"X": 0.0, "Y": 0.0, "Z": 0.0},
                        "EndPoint": {"X": 5.0, "Y": 0.0, "Z": 0.0},
                        "MidPoint": {"X": 2.5, "Y": 0.0, "Z": 0.0},
                        "Center": {"X": 2.5, "Y": 0.0, "Z": 0.0},
                        "Radius": 2.5,
                        "StartAngle": 0.0,
                        "EndAngle": 180.0,
                        "ColorIndex": 1,
                        "CurveType": 1
                    }]
                }],
                "jcws": [{
                    "GuidName": "jcw-001",
                    "Type": 1,
                    "Name": "JCW1",
                    "CenterPoint": {"X": 1.0, "Y": 1.0, "Z": 1.0},
                    "BoundaryLines": [{
                        "StartPoint": {"X": 0.0, "Y": 0.0, "Z": 0.0},
                        "EndPoint": {"X": 2.0, "Y": 2.0, "Z": 2.0}
                    }]
                }],
                "door": [{
                    "Guid": "door-001",
                    "FamilyName": "Door1",
                    "DoorType": 1,
                    "HostWall": "wall-001",
                    "Location": {"X": 1.0, "Y": 1.0, "Z": 1.0},
                    "Size": {
                        "Height": 2.1,
                        "Width": 0.9,
                        "Thickness": 0.1
                    },
                    "FlipFaceNormal": {"X": 0.0, "Y": 1.0, "Z": 0.0},
                    "FlipHandNormal": {"X": 1.0, "Y": 0.0, "Z": 0.0}
                }]
            }
        }]
    })";

    ARDesign result = parseARDesign(completeJson);

    // Test Floor basic info
    ASSERT_EQ(result.Floor.size(), 1);
    ASSERT_EQ(result.Floor[0].Name, "F1");

    // Test HouseType
    const auto &houseType = result.Floor[0].construction.houseTypes[0];
    ASSERT_EQ(houseType.houseName, "House1");
    ASSERT_DOUBLE_EQ(houseType.Boundary[0].x, 1.0);

    // Test Room
    const auto &room = result.Floor[0].construction.rooms[0];
    ASSERT_EQ(room.Name, "LivingRoom");
    ASSERT_EQ(room.Boundary[0].CurveType, 1);
    ASSERT_DOUBLE_EQ(room.Boundary[0].Radius, 2.5);

    // Test JCW
    const auto &jcw = result.Floor[0].construction.jcws[0];
    ASSERT_EQ(jcw.GuidName, "jcw-001");
    ASSERT_DOUBLE_EQ(jcw.CenterPoint.x, 1.0);

    // Test Door
    const auto &door = result.Floor[0].construction.door[0];
    ASSERT_EQ(door.Guid, "door-001");
    ASSERT_DOUBLE_EQ(door.Size.Height, 2.1);
    ASSERT_DOUBLE_EQ(door.FlipFaceNormal.y, 1.0);
}

// Test parsing actual ARDesign-min.json file
TEST_F(ParseARDesignTest, ParsesActualARDesignFile) {
    // 获取当前工作目录（帮助调试）
    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));
    // Read file content
    // std::ifstream file("../example/ARDesign-min.json");
    std::string filePath = "../example/ARDesign-min.json";
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
    ARDesign result = parseARDesign(json);

    // Test basic structure
    ASSERT_EQ(result.Floor.size(), 1);
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

}
