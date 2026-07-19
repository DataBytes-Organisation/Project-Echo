import express from "express";

const router = express.Router();

router.get("/movement_time_daily/:start/:end", async (req, res) => {
  const start = Number(req.params.start);
  const end   = Number(req.params.end);

  const rows = await req.db.collection("events").aggregate([
    { $match: { ts: { $gte: start, $lt: end } } },
    {
      $group: {
        _id: {
          day: {
            $dateToString: {
              format: "%Y-%m-%d",
              date: { $toDate: { $multiply: ["$ts", 1000] } },
              timezone: "Australia/Perth"
            }
          }
        },
        count: { $sum: 1 }
      }
    },
    { $sort: { "_id.day": 1 } },
    { $project: { _id: 0, day: "$_id.day", count: 1 } }
  ]).toArray();

  res.json(rows);
});

export default router;
