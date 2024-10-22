// tests/test_helper.cpp
#include <gtest/gtest.h>
#include "../include/helper.hpp"
#include "../include/types/heating_design_structures.hpp"
#include <sstream>
#include <json/json.h>


TEST(HelperTest, ParseJsonData) {
    // 准备测试数据
    std::string arDesignJson = R"({
        "Floors": [{
            "Name": "Floor1",
            "Num": "1",
            "LevelHeight": 3.0,
            "Construction": {
                "HouseType": [{
                    "houseName": "Type A",
                    "RoomNames": ["Living Room", "Bedroom"],
                    "Boundary": [{"x": 0, "y": 0, "z": 0}, {"x": 10, "y": 10, "z": 0}]
                }],
                "Room": [{
                    "Guid": "room1",
                    "Name": "Living Room",
                    "NameType": "LivingRoom",
                    "DoorIds": ["door1"],
                    "JCWGuidNames": ["jcw1"],
                    "WallNames": ["wall1"],
                    "Boundary": [{"x": 0, "y": 0, "z": 0}, {"x": 5, "y": 5, "z": 0}],
                    "IsRecreationalRoom": false
                }]
            }
        }]
    })";

    std::string inputDataJson = R"({
        "AssistData": {
            "AssistCollectors": [{
                "Id": "collector1",
                "Loc": {"x": 1, "y": 1, "z": 0},
                "LevelName": "Floor1",
                "Boundaries": [{
                    "Offset": 0.1,
                    "Borders": [{
                        "StartPoint": {"x": 0, "y": 0, "z": 0},
                        "EndPoint": {"x": 1, "y": 1, "z": 0},
                        "ColorIndex": 1,
                        "CurveType": 0
                    }]
                }]
            }]
        },
        "WebData": {
            "ImbalanceRatio": 10,
            "JointPipeSpan": 0.5,
            "DenseAreaWallSpan": 0.3,
            "DenseAreaSpanLess": 0.2,
            "LoopSpanSet": [{
                "TypeName": "Type1",
                "MinSpan": 0.1,
                "MaxSpan": 0.5,
                "Curvity": 0.2
            }],
            "ObsSpanSet": [{
                "ObsName": "Obstacle1",
                "MinSpan": 0.1,
                "MaxSpan": 0.3
            }],
            "DeliverySpanSet": [{
                "ObsName": "Delivery1",
                "MinSpan": 0.2,
                "MaxSpan": 0.4
            }],
            "PipeSpanSet": [{
                "LevelDesc": "Floor1",
                "FuncName": "Function1",
                "Directions": ["North", "South"],
                "ExterWalls": 2,
                "PipeSpan": 0.3
            }],
            "ElasticSpanSet": [{
                "FuncName": "Function1",
                "PriorSpan": 0.2,
                "MinSpan": 0.1,
                "MaxSpan": 0.3
            }],
            "FuncRooms": [{
                "FuncName": "Function1",
                "RoomNames": ["Living Room", "Bedroom"]
            }]
        }
    })";

    // 调用被测试的函数
    CombinedData result = parseJsonData(arDesignJson, inputDataJson);

    // 验证结果
    EXPECT_EQ(result.arDesign.Floors.size(), 1);
    EXPECT_EQ(result.arDesign.Floors[0].Name, "Floor1");
    EXPECT_EQ(result.arDesign.Floors[0].Num, "1");
    EXPECT_DOUBLE_EQ(result.arDesign.Floors[0].LevelHeight, 3.0);

    EXPECT_EQ(result.inputData.assistData.AssistCollectors.size(), 1);
    EXPECT_EQ(result.inputData.assistData.AssistCollectors[0].Id, "collector1");

    EXPECT_EQ(result.inputData.webData.ImbalanceRatio, 10);
    EXPECT_DOUBLE_EQ(result.inputData.webData.JointPipeSpan, 0.5);
    EXPECT_EQ(result.inputData.webData.LoopSpanSet.size(), 1);
    EXPECT_EQ(result.inputData.webData.ObsSpanSet.size(), 1);
    EXPECT_EQ(result.inputData.webData.DeliverySpanSet.size(), 1);
    EXPECT_EQ(result.inputData.webData.PipeSpanSet.size(), 1);
    EXPECT_EQ(result.inputData.webData.ElasticSpanSet.size(), 1);
    EXPECT_EQ(result.inputData.webData.FuncRooms.size(), 1);
}

TEST(HelperTest, PlanToJson) {
    // 准备测试数据
    HeatingDesign plan;
    HeatingCoil heatingCoil;
    heatingCoil.LevelName = "Floor1";
    heatingCoil.LevelNo = 1;
    heatingCoil.LevelDesc = "First Floor";
    heatingCoil.HouseName = "House A";

    JLine expansion;
    expansion.StartPoint = {0, 0, 0};
    expansion.EndPoint = {1, 1, 1};
    expansion.ColorIndex = 1;
    expansion.CurveType = 0;
    heatingCoil.Expansions.push_back(expansion);

    CollectorCoil collectorCoil;
    collectorCoil.CollectorName = "Collector1";
    collectorCoil.Loops = 2;

    CoilLoop coilLoop;
    coilLoop.Length = 10.0;
    coilLoop.Curvity = 2;
    coilLoop.Areas.push_back({"Area1"});
    coilLoop.Path.push_back({
        {0, 0, 0},
        {1, 1, 1},
        1,
        0
    });
    collectorCoil.CoilLoops.push_back(coilLoop);

    collectorCoil.Deliverys.push_back({{
        {0, 0, 0},
        {1, 1, 1},
        1,
        0
    }});

    heatingCoil.CollectorCoils.push_back(collectorCoil);
    plan.HeatingCoils.push_back(heatingCoil);

    // 调用被测试的函数
    std::string result = planToJson(plan);

    // 验证结果
    Json::Value root;
    Json::Reader reader;
    ASSERT_TRUE(reader.parse(result, root));

    EXPECT_EQ(root["HeatingCoils"].size(), 1);
    EXPECT_EQ(root["HeatingCoils"][0]["LevelName"].asString(), "Floor1");
    EXPECT_EQ(root["HeatingCoils"][0]["LevelNo"].asInt(), 1);
    EXPECT_EQ(root["HeatingCoils"][0]["LevelDesc"].asString(), "First Floor");
    EXPECT_EQ(root["HeatingCoils"][0]["HouseName"].asString(), "House A");

    EXPECT_EQ(root["HeatingCoils"][0]["Expansions"].size(), 1);
    EXPECT_EQ(root["HeatingCoils"][0]["CollectorCoils"].size(), 1);
    EXPECT_EQ(root["HeatingCoils"][0]["CollectorCoils"][0]["CollectorName"].asString(), "Collector1");
    EXPECT_EQ(root["HeatingCoils"][0]["CollectorCoils"][0]["Loops"].asInt(), 2);
    EXPECT_EQ(root["HeatingCoils"][0]["CollectorCoils"][0]["CoilLoops"].size(), 1);
    EXPECT_EQ(root["HeatingCoils"][0]["CollectorCoils"][0]["Deliverys"].size(), 1);
}

TEST(HelperTest, GeneratePipePlan) {
    // 准备测试数据
    CombinedData combinedData;
    Floor floor;
    floor.Name = "Floor1";
    floor.Num = "1";
    floor.LevelHeight = 3.0;

    HouseType houseType;
    houseType.houseName = "Type A";
    houseType.RoomNames = {"Living Room", "Bedroom"};
    houseType.Boundary = {{0, 0, 0}, {10, 10, 0}};
    floor.construction.houseTypes.push_back(houseType);

    combinedData.arDesign.Floors.push_back(floor);

    combinedData.inputData.webData.ImbalanceRatio = 10;
    combinedData.inputData.webData.JointPipeSpan = 0.5;
    combinedData.inputData.webData.DenseAreaWallSpan = 0.3;
    combinedData.inputData.webData.DenseAreaSpanLess = 0.2;

    // 调用被测试的函数
    HeatingDesign result = generatePipePlan(combinedData);

    // 验证结果
    EXPECT_EQ(result.HeatingCoils.size(), 1);
    EXPECT_EQ(result.HeatingCoils[0].LevelName, "Floor1");
    EXPECT_EQ(result.HeatingCoils[0].LevelNo, 1);
    EXPECT_EQ(result.HeatingCoils[0].LevelDesc, "Floor 1");
    EXPECT_EQ(result.HeatingCoils[0].HouseName, "Type A");
    EXPECT_GT(result.HeatingCoils[0].CollectorCoils.size(), 0);
}

