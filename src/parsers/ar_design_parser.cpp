#include <iostream>
#include "parsers/ar_design_parser.hpp"

namespace iad {
namespace parsers {

void ARDesignParser::parseCurveInfo(const Json::Value& curveJson, CurveInfo& curve) {
    // Parse StartPoint
    const Json::Value& startPointJson = curveJson["StartPoint"];
    curve.StartPoint = Point{
        startPointJson["x"].asDouble(),
        startPointJson["y"].asDouble(),
        startPointJson["z"].asDouble()
    };

    // Parse EndPoint
    const Json::Value& endPointJson = curveJson["EndPoint"];
    curve.EndPoint = Point{
        endPointJson["x"].asDouble(),
        endPointJson["y"].asDouble(),
        endPointJson["z"].asDouble()
    };

    // Parse optional fields if they exist
    if (curveJson.isMember("MidPoint")) {
        const Json::Value& midPointJson = curveJson["MidPoint"];
        curve.MidPoint = Point{
            midPointJson["x"].asDouble(),
            midPointJson["y"].asDouble(),
            midPointJson["z"].asDouble()
        };
    }

    if (curveJson.isMember("Center")) {
        const Json::Value& centerJson = curveJson["Center"];
        curve.Center = Point{
            centerJson["x"].asDouble(),
            centerJson["y"].asDouble(),
            centerJson["z"].asDouble()
        };
    }

    curve.Radius = curveJson.get("Radius", 0.0).asDouble();
    curve.StartAngle = curveJson.get("StartAngle", 0.0).asDouble();
    curve.EndAngle = curveJson.get("EndAngle", 0.0).asDouble();
    curve.ColorIndex = curveJson.get("ColorIndex", 0).asInt();
    curve.CurveType = curveJson.get("CurveType", 0).asInt();
}

void ARDesignParser::parseRoom(const Json::Value& roomJson, Room& room) {
    room.Name = roomJson.get("Name", "").asString();
    room.Guid = roomJson.get("Guid", "").asString();
    room.NameType = roomJson.get("NameType", "").asString();
    room.BlCreateRoom = roomJson.get("BlCreateRoom", 0).asInt();

    // Parse DoorIds
    if (roomJson.isMember("DoorIds") && roomJson["DoorIds"].isArray()) {
        for (const auto& doorId : roomJson["DoorIds"]) {
            room.DoorIds.push_back(doorId.asString());
        }
    }

    // Parse JCWGuidNames
    if (roomJson.isMember("JCWGuidNames") && roomJson["JCWGuidNames"].isArray()) {
        for (const auto& jcwGuidName : roomJson["JCWGuidNames"]) {
            room.JCWGuidNames.push_back(jcwGuidName.asString());
        }
    }

    // Parse WallNames
    if (roomJson.isMember("WallNames") && roomJson["WallNames"].isArray()) {
        for (const auto& wallName : roomJson["WallNames"]) {
            room.WallNames.push_back(wallName.asString());
        }
    }

    // Parse Borders if they exist
    if (roomJson.isMember("Boundary") && roomJson["Boundary"].isArray()) {
        for (const auto& borderJson : roomJson["Boundary"]) {
            CurveInfo border;
            border.StartPoint = Point{
                borderJson["StartPoint"]["x"].asDouble(),
                borderJson["StartPoint"]["y"].asDouble(),
                borderJson["StartPoint"]["z"].asDouble()
            };
            border.EndPoint = Point{
                borderJson["EndPoint"]["x"].asDouble(),
                borderJson["EndPoint"]["y"].asDouble(),
                borderJson["EndPoint"]["z"].asDouble()
            };
            border.ColorIndex = borderJson.get("ColorIndex", 0).asInt();
            border.CurveType = borderJson.get("CurveType", 0).asInt();
            room.Boundary.push_back(border);
        }
    }
}

void ARDesignParser::parseJCW(const Json::Value& jcwJson, JCW& jcw) {
    // Parse basic properties
    jcw.GuidName = jcwJson.get("GuidName", "").asString();
    jcw.Type = jcwJson.get("Type", 0).asInt();
    jcw.Name = jcwJson.get("Name", "").asString();

    // Parse CenterPoint
    if (jcwJson.isMember("CenterPoint")) {
        const auto& centerPoint = jcwJson["CenterPoint"];
        jcw.CenterPoint = Point{
            centerPoint["x"].asDouble(),
            centerPoint["y"].asDouble(),
            centerPoint["z"].asDouble()
        };
    }

    // Parse BoundaryLines array
    if (jcwJson.isMember("BoundaryLines") && jcwJson["BoundaryLines"].isArray()) {
        for (const auto& lineJson : jcwJson["BoundaryLines"]) {
            CurveInfo curve;
            parseCurveInfo(lineJson, curve);
            jcw.BoundaryLines.push_back(curve);
        }
    }
}

void ARDesignParser::parseDoor(const Json::Value& doorJson, Door& door) {
    // Parse basic properties
    door.Guid = doorJson.get("Guid", "").asString();
    door.FamilyName = doorJson.get("FamilyName", "").asString();
    door.DoorType = doorJson.get("DoorType", "").asString();
    door.HostWall = doorJson.get("HostWall", "").asString();

    // Parse Location
    if (doorJson.isMember("Location")) {
        const auto& location = doorJson["Location"];
        door.Location = Point{
            location["x"].asDouble(),
            location["y"].asDouble(),
            location["z"].asDouble()
        };
    }

    // Parse Size
    if (doorJson.isMember("Size")) {
        const auto& size = doorJson["Size"];
        door.Size = Size{
            size.get("Height", 0.0).asDouble(),
            size.get("Width", 0.0).asDouble(),
            size.get("Thickness", 0.0).asDouble()
        };
    }

    // Parse FlipFaceNormal
    if (doorJson.isMember("FlipFaceNormal")) {
        const auto& flipFace = doorJson["FlipFaceNormal"];
        door.FlipFaceNormal = Point{
            flipFace["x"].asDouble(),
            flipFace["y"].asDouble(),
            flipFace["z"].asDouble()
        };
    }

    // Parse FlipHandNormal
    if (doorJson.isMember("FlipHandNormal")) {
        const auto& flipHand = doorJson["FlipHandNormal"];
        door.FlipHandNormal = Point{
            flipHand["x"].asDouble(),
            flipHand["y"].asDouble(),
            flipHand["z"].asDouble()
        };
    }
}

void ARDesignParser::parseHouseType(const Json::Value& houseTypeJson, HouseType& houseType) {
    // Parse basic properties
    houseType.houseName = houseTypeJson.get("houseName", "").asString();

    // Parse RoomNames array
    if (houseTypeJson.isMember("RoomNames") && houseTypeJson["RoomNames"].isArray()) {
        for (const auto& roomName : houseTypeJson["RoomNames"]) {
            houseType.RoomNames.push_back(roomName.asString());
        }
    }

    // Parse Boundary array
    if (houseTypeJson.isMember("Boundary") && houseTypeJson["Boundary"].isArray()) {
        for (const auto& pointJson : houseTypeJson["Boundary"]) {
            Point point{
                pointJson["x"].asDouble(),
                pointJson["y"].asDouble(),
                pointJson["z"].asDouble()
            };
            houseType.Boundary.push_back(point);
        }
    }
}

ARDesign ARDesignParser::parse(const std::string& jsonStr) {
    ARDesign design;
    Json::Value j;
    Json::Reader reader;
    
    if (!reader.parse(jsonStr, j)) {
        throw std::runtime_error("Failed to parse AR Design JSON");
    }
    
    try {
        // Parse Floor array
        if (j.isMember("Floor") && j["Floor"].isArray()) {
            for (const auto& floorJson : j["Floor"]) {
                Floor floor;
                
                // Add debug output
                // std::cout << "Parsing floor: " << floorJson.toStyledString() << std::endl;
                
                // Parse floor basic info
                floor.Name = floorJson.get("Name", "").asString();
                floor.Num = floorJson.get("Num", "").asString();
                floor.LevelHeight = floorJson.get("LevelHeight", 0.0).asDouble();
                
                // Parse Construction
                if (floorJson.isMember("Construction")) {
                    std::cout << "Found Construction field" << std::endl;
                    const auto& constJson = floorJson["Construction"];
                    
                    // Parse HouseTypes
                    if (constJson.isMember("HouseType")) {
                        std::cout << "Found HouseTypes, count: " << constJson["HouseType"].size() << std::endl;
                        for (const auto& houseTypeJson : constJson["HouseType"]) {
                            HouseType houseType;
                            parseHouseType(houseTypeJson, houseType);
                            floor.construction.houseTypes.push_back(houseType);
                        }
                    } else {
                        std::cout << "No HouseTypes field found" << std::endl;
                    }
                    
                    // Parse Rooms
                    if (constJson.isMember("Room")) {
                        std::cout << "Found Rooms, count: " << constJson["Room"].size() << std::endl;
                        for (const auto& roomJson : constJson["Room"]) {
                            Room room;
                            parseRoom(roomJson, room);
                            floor.construction.rooms.push_back(room);
                        }
                    } else {
                        std::cout << "No Rooms field found" << std::endl;
                    }
                    
                    // Parse JCWs
                    if (constJson.isMember("JCW")) {
                        const auto& jcws = constJson["JCW"];
                        std::cout << "Found JCWs field, count: " << jcws.size() << std::endl;
                        for (const auto& jcwJson : jcws) {
                            JCW jcw;
                            parseJCW(jcwJson, jcw);
                            floor.construction.jcws.push_back(jcw);
                        }
                    } else {
                        std::cout << "No JCWs field found" << std::endl;
                    }

                    // Parse Doors
                    if (constJson.isMember("Door")) {
                        const auto& doors = constJson["Door"];
                        std::cout << "Found Doors field, count: " << doors.size() << std::endl;
                        for (const auto& doorJson : doors) {
                            Door door;
                            parseDoor(doorJson, door);
                            floor.construction.door.push_back(door);
                        }
                    } else {
                        std::cout << "No Doors field found" << std::endl;
                    }

                    // After parsing all Construction data
                    std::cout << "Construction parsing complete:" << std::endl;
                    std::cout << "  HouseTypes count: " << floor.construction.houseTypes.size() << std::endl;
                    std::cout << "  Rooms count: " << floor.construction.rooms.size() << std::endl;
                    std::cout << "  JCWs count: " << floor.construction.jcws.size() << std::endl;
                    std::cout << "  Doors count: " << floor.construction.door.size() << std::endl;
                } else {
                    std::cout << "No Construction field found" << std::endl;
                }
                design.Floor.push_back(floor);
            }
        }
    } catch (const Json::Exception& e) {
        std::cerr << "JSON parsing error in parseARDesign: " << e.what() << std::endl;
        throw;
    }
    return design;
}

} // namespace parsers
} // namespace iad 