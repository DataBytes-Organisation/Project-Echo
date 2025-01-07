import React from 'react';
import './uex_leadership_card.css';

const UEXLeadershipCard = ({ metrics, leaderboard }) => {
    return (
        <div className="uex-card">
            <h2>User Experience Leadership</h2>
            <div className="metrics">
                <p>Total Sightings Uploaded: {metrics.totalSightings}</p>
                <p>Average Upload Time: {metrics.avgUploadTime}</p>
                <p>User Satisfaction Score: {metrics.satisfactionScore}%</p>
            </div>
            <div className="leaderboard">
                <h3>Top Contributors</h3>
                <ul>
                    {leaderboard.map((user, index) => (
                        <li key={index}>
                            {user.name} - {user.uploads} uploads
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
};

export default UEXLeadershipCard;
