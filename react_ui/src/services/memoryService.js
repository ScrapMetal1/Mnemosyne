/* src/services/memoryService.js */ 
const API_BASE = '/api' 
 
export const memoryService = {  // this is memory related tasks package 
    async captureMemory(imageBase64) { 
        try { 
            const response = await fetch(`${API_BASE}/memory/capture`, { 
                method: "POST", 
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageBase64 }),
            }) 
 
            if (!response.ok) { 
                throw new Error(`Memory capture failed: ${response.statusText}`) 
            } 
 
            const data = await response.json() 
            return data 
        } catch (error) { 
            console.error("Error capturing memory:", error) 
            throw error // so the component know its failed. Not really necessary here.  
        } 
    } 
} 
