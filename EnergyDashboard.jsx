import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function EnergyDashboard() {
  const [currentMonth, setCurrentMonth] = useState('April 2025');
  
  // Generate 24 hours of data similar to the Python code
  const generateData = () => {
    const hours = Array.from({ length: 24 }, (_, i) => i);
    
    // Add noise function
    const addNoise = (base, strength = 0.2) => {
      return base.map(val => val + strength * (Math.random() * 2 - 1));
    };
    
    // Temperature data
    const freshTempBase = hours.map(h => 15 + 5 * Math.sin(2 * Math.PI * (h - 14) / 24));
    const freshTemp = addNoise(freshTempBase, 0.5);
    
    const salineTempBase = hours.map(h => 18 + 4 * Math.sin(2 * Math.PI * (h - 12) / 24));
    const salineTemp = addNoise(salineTempBase, 0.6);
    
    // Pressure data
    const freshPressureBase = hours.map(h => 2.5 + 0.3 * Math.sin(2 * Math.PI * h / 24));
    const freshPressure = addNoise(freshPressureBase, 0.1);
    
    const salinePressureBase = hours.map(h => 7.2 + 0.8 * Math.sin(2 * Math.PI * h / 12) + 0.4 * Math.sin(2 * Math.PI * h / 6));
    const salinePressure = addNoise(salinePressureBase, 0.2);
    
    // Salinity data
    const freshSalinityBase = hours.map(h => 0.5 + 0.1 * Math.sin(2 * Math.PI * h / 24));
    const freshSalinity = addNoise(freshSalinityBase, 0.05).map(val => Math.max(val, 0.3));
    
    const salineSalinityBase = hours.map(h => 35 + 2 * Math.sin(2 * Math.PI * (h - 3) / 12));
    const salineSalinity = addNoise(salineSalinityBase, 0.8);
    
    // Energy output
    const energyRegular = hours.map((_, i) => {
      const pressureDiff = salinePressure[i] - freshPressure[i];
      const salinityDiff = salineSalinity[i] - freshSalinity[i];
      const tempFactor = 1 / (Math.abs(salineTemp[i] - freshTemp[i]) * 0.1 + 1);
      return pressureDiff * salinityDiff * tempFactor * 0.2;
    });
    const finalEnergyRegular = addNoise(energyRegular, 0.8);
    
    // QML optimized energy
    const theoreticalMax = Math.max(...finalEnergyRegular) * 1.3;
    const baseImprovement = finalEnergyRegular.map(val => val * 1.02);
    const energyQml = baseImprovement.map(val => {
      const distanceToMax = theoreticalMax - val;
      return val + (distanceToMax * 0.4);
    });
    const finalEnergyQml = addNoise(energyQml, 0.2);
    
    // Format data for charts
    return hours.map((hour, i) => ({
      name: `${hour}:00`,
      regularOutput: finalEnergyRegular[i],
      qmlOutput: finalEnergyQml[i],
      freshTemp: freshTemp[i],
      salineTemp: salineTemp[i],
      freshPressure: freshPressure[i],
      salinePressure: salinePressure[i],
      freshSalinity: freshSalinity[i],
      salineSalinity: salineSalinity[i]
    }));
  };
  
  const data = generateData();
  
  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      {/* Navbar */}
      <nav className="flex items-center justify-between p-4 bg-gray-900 border-b border-gray-800">
        <div className="flex items-center">
          <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center mr-2">
            <span className="text-2xl font-bold">Q</span>
          </div>
          <span className="text-lg font-semibold text-blue-300">QBLUE</span>
        </div>
        
        <h1 className="text-xl font-bold">Energy Output Analytics</h1>
        
        <div className="relative">
          <button className="bg-purple-500 text-white px-4 py-2 rounded flex items-center">
            {currentMonth}
            <svg className="ml-2 h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
            </svg>
          </button>
        </div>
      </nav>
      
      {/* Sidebar */}
      <div className="flex flex-1 overflow-hidden">
        <aside className="w-48 bg-gray-900 border-r border-gray-800 flex flex-col">
          <div className="p-4">
            <div className="relative">
              <input type="text" placeholder="Search for..." className="w-full bg-gray-800 text-white rounded px-4 py-2 pl-8" />
              <svg className="absolute left-2 top-2.5 h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
              </svg>
            </div>
          </div>
          
          <div className="px-4 py-2 flex items-center text-purple-400">
            <span className="mr-2">
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
              </svg>
            </span>
            Sensor Data
          </div>
          
          <div className="px-4 py-2 bg-gray-800 text-white font-medium">
            Energy Output
          </div>
          
          <div className="px-4 py-2 text-gray-400">
            Live Feed
          </div>
          
          <div className="mt-auto p-4">
            <div className="flex items-center text-gray-400 mb-4">
              <svg className="h-5 w-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
              </svg>
              Settings
            </div>
            
            <div className="flex items-center">
              <div className="w-8 h-8 bg-purple-600 rounded-full mr-2 flex items-center justify-center text-white font-bold">
                J
              </div>
              <div>
                <div className="font-medium">John Carter</div>
                <div className="text-xs text-gray-400">Account settings</div>
              </div>
            </div>
          </div>
        </aside>
        
        {/* Main content */}
        <main className="flex-1 overflow-y-auto p-6">
          {/* Energy Output Chart */}
          <div className="bg-gray-800 rounded-lg p-4 mb-6">
            <h2 className="text-lg font-semibold mb-4">Energy Output Comparison</h2>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#3d2c63" />
                  <XAxis 
                    dataKey="name" 
                    stroke="#ffffff" 
                    label={{ value: 'Time (Hours)', position: 'bottom', fill: '#ffffff' }} 
                  />
                  <YAxis 
                    stroke="#ffffff" 
                    label={{ value: 'Energy Output (MW)', angle: -90, position: 'left', fill: '#ffffff' }}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#2d1e4f', borderColor: '#3d2c63', color: '#ffffff' }} 
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="regularOutput" 
                    name="Regular Output" 
                    stroke="#e74c3c" 
                    activeDot={{ r: 8 }} 
                    strokeWidth={3}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="qmlOutput" 
                    name="QML Optimized" 
                    stroke="#9b59b6" 
                    activeDot={{ r: 8 }} 
                    strokeWidth={3}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Parameter Grids */}
          <div className="bg-blue-900 bg-opacity-30 rounded-lg p-4 mb-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-center font-bold uppercase">Temperature</h3>
              <h3 className="text-center font-bold uppercase">Pressure</h3>
              <h3 className="text-center font-bold uppercase">Salinity</h3>
            </div>
            
            <div className="grid grid-cols-3 gap-4">
              {/* Temperature charts */}
              <div className="bg-gray-800 rounded p-2">
                <div style={{ height: "120px" }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#3d2c63" />
                      <XAxis dataKey="name" tick={false} />
                      <YAxis />
                      <Line 
                        type="monotone" 
                        dataKey="salineTemp" 
                        stroke="#0d47a1" 
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              {/* Pressure charts */}
              <div className="bg-gray-800 rounded p-2">
                <div style={{ height: "120px" }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#3d2c63" />
                      <XAxis dataKey="name" tick={false} />
                      <YAxis />
                      <Line 
                        type="monotone" 
                        dataKey="salinePressure" 
                        stroke="#004d40" 
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              {/* Salinity charts */}
              <div className="bg-gray-800 rounded p-2">
                <div style={{ height: "120px" }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#3d2c63" />
                      <XAxis dataKey="name" tick={false} />
                      <YAxis />
                      <Line 
                        type="monotone" 
                        dataKey="freshSalinity" 
                        stroke="#64b5f6" 
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-blue-900 bg-opacity-30 rounded-lg p-4">
            <div className="flex items-center mb-4">
              <div className="flex-shrink-0 w-8 font-medium text-center text-blue-300 mr-2">
                <div>F</div>
                <div>R</div>
                <div>E</div>
                <div>S</div>
                <div>H</div>
              </div>
              
              <div className="grid grid-cols-3 gap-4 flex-grow">
                {/* Fresh temperature chart */}
                <div className="bg-gray-800 rounded p-2">
                  <div style={{ height: "120px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#3d2c63" />
                        <XAxis dataKey="name" tick={false} />
                        <YAxis />
                        <Line 
                          type="monotone" 
                          dataKey="freshTemp" 
                          stroke="#1976d2" 
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                
                {/* Fresh pressure chart */}
                <div className="bg-gray-800 rounded p-2">
                  <div style={{ height: "120px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#3d2c63" />
                        <XAxis dataKey="name" tick={false} />
                        <YAxis />
                        <Line 
                          type="monotone" 
                          dataKey="freshPressure" 
                          stroke="#00796b" 
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                
                {/* Fresh salinity chart */}
                <div className="bg-gray-800 rounded p-2">
                  <div style={{ height: "120px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#3d2c63" />
                        <XAxis dataKey="name" tick={false} />
                        <YAxis />
                        <Line 
                          type="monotone" 
                          dataKey="freshSalinity" 
                          stroke="#4db6ac" 
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex items-center">
              <div className="flex-shrink-0 w-8 font-medium text-center text-blue-300 mr-2">
                <div>S</div>
                <div>A</div>
                <div>L</div>
                <div>I</div>
                <div>N</div>
                <div>E</div>
              </div>
              
              <div className="grid grid-cols-3 gap-4 flex-grow">
                {/* Saline temperature chart */}
                <div className="bg-gray-800 rounded p-2">
                  <div style={{ height: "120px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#3d2c63" />
                        <XAxis dataKey="name" tick={false} />
                        <YAxis />
                        <Line 
                          type="monotone" 
                          dataKey="salineTemp" 
                          stroke="#0d47a1" 
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                
                {/* Saline pressure chart */}
                <div className="bg-gray-800 rounded p-2">
                  <div style={{ height: "120px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#3d2c63" />
                        <XAxis dataKey="name" tick={false} />
                        <YAxis />
                        <Line 
                          type="monotone" 
                          dataKey="salinePressure" 
                          stroke="#004d40" 
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                
                {/* Saline salinity chart */}
                <div className="bg-gray-800 rounded p-2">
                  <div style={{ height: "120px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#3d2c63" />
                        <XAxis dataKey="name" tick={false} />
                        <YAxis />
                        <Line 
                          type="monotone" 
                          dataKey="salineSalinity" 
                          stroke="#00796b" 
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}